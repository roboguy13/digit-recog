{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE QuantifiedConstraints #-}

-- {-# OPTIONS_GHC -Wall #-}


module Backprop (backprop, Net, NetState, computeNetState) where

import           Prelude hiding (unzip)

import           Control.Lens.At
import           Control.Lens.Indexed
import           Control.Lens ((^?))

import           Data.Traversable

import           Linear.Vector
import           Linear.Trace hiding (trace)
import           Linear.Matrix hiding (trace)
import           Linear.Metric

import           Numeric.AD.Mode.Reverse

import           Dense
import           Utils


import Debug.Trace


type Net      f a = f (Dense f a)
type NetState f a = f (f (NeuronState f a))

-- Feedforward
computeNetState :: (Traversable f, Metric f, Floating a, Ord a, Index (Net f a) ~ Int, IxValue (Net f a) ~ Dense f a, Ixed (Net f a)) =>
  Net f a -> f a -> NetState f a
computeNetState net inputs =
    snd $ mapAccumL go Nothing net
  where
    go Nothing currLayer =
      let currOutputs = denseOutput currLayer inputs
      in (Just currOutputs, currOutputs)

    go (Just prevOutputs) currLayer =
      let currOutputs = denseOutput currLayer (fmap neuronStateOutput prevOutputs)
      in (Just currOutputs, currOutputs)

backprop :: forall f a.  (Show a, LayerCtx f a) =>
  Int -> a -> f (f a, f a) -> f (f (Neuron f a)) -> f (Dense f a)
backprop minibatchSize stepSize minibatch layers =
  zipWithTF processOneLayer cwGrads layersAndDeltas
  where
    -- | Indexed by mini-batch index
    netStates :: f (NetState f a)
    netStates = fmap (computeNetState layers . fst) minibatch

    processOneLayer ::
      f (f (f a)) -> (f (Neuron f a), f (f a)) ->
      f (Neuron f a)
    processOneLayer grads (neuronStates, currDeltas) =
      zipWithTF processNeuron (transpose' grads) (zipTF neuronStates (transpose' currDeltas))
    {-# INLINE processOneLayer #-}

    processNeuron :: f (f a) -> (Neuron f a, f a) -> Neuron f a
    processNeuron grads (neuron, delta) =
      neuron
        { neuronWeights =
            neuronWeights neuron ^-^ ((stepSize / fromIntegral minibatchSize) *^ weightsSum)
        , neuronBias    =
            neuronBias neuron - ((stepSize / fromIntegral minibatchSize) * biasSum)
        }
      where
        weightsSum =
          foldr (^+^) zero grads
        {-# INLINE weightsSum #-}

        biasSum =
          sum delta
        {-# INLINE biasSum #-}
    {-# INLINE processNeuron #-}

    layersAndDeltas :: f (f (Neuron f a), f (f a))
    layersAndDeltas =
      zipTF layers deltas
    {-# INLINE layersAndDeltas #-}

    -- | Indexed by mini-batch index, then by layer, then by incoming neuron, then by outgoing neuron
    -- (?)
    cwGrads :: f (f (f (f a)))
    cwGrads = transpose' $
      zipWithTF
        (\(currInputs, netState) currDeltas -> imap (findCWGrad currInputs netState) currDeltas)
        (zipTF (fmap fst minibatch) netStates)
        (transpose' deltas)
    {-# INLINE cwGrads #-}

    -- Indexed first by mini-batch index
    deltas :: f (f (f a))
    deltas =
      transpose' $
      zipWithTF (\(currInputs, currExpected) netState ->
                    snd $ mapAccumR (findDeltas currExpected) Nothing netState)
                minibatch
                netStates

    findDeltas :: f a
      -> Maybe (f (NeuronState f a), f a)
      -> f (NeuronState f a)
      -> (Maybe (f (NeuronState f a), f a), f a)
    findDeltas currExpected maybeNext currLayer =
      let currDeltas = computeDeltas currExpected currLayer maybeNext
      in (Just (currLayer, currDeltas), currDeltas)
    {-# INLINE findDeltas #-}

    findCWGrad :: f a -> NetState f a -> Int -> f a -> f (f a)
    findCWGrad currInputs netState currIx currDeltas =
      costWeightGrad currInputs currDeltas
                     (netState ^? ix (currIx-1))
    {-# INLINE findCWGrad #-}

-- | In the result: The outermost "container" is indexed by incoming neuron
-- index (j) and the innermost "container" is indexed by the outgoing neuron
-- index (k)      (?)
costWeightGrad :: forall f a. LayerCtx f a =>
  f a -> f a ->
  Maybe (f (NeuronState f a)) ->
  f (f a)
costWeightGrad inputs currDeltas maybePrevLayer =
  outerInlined currDeltas prevA
  where
    prevA =
      case maybePrevLayer of
        Nothing        -> inputs
        Just prevLayer -> fmap neuronStateOutput prevLayer
        -- Just prevLayer -> fmap neuronStateOutput $ denseOutput prevLayer inputs
        -- Just prevLayer -> fmap neuronStateOutput prevLayer
{-# INLINE costWeightGrad #-}

-- | `maybeNext` is the next layer together with the deltas from the next
-- layer. `expected` is the expected outputs.
--
-- The result is the deltas for the given layer.
computeDeltas :: forall f a. LayerCtx f a =>
  f a ->
  f (NeuronState f a) -> Maybe (f (NeuronState f a), f a) -> f a
computeDeltas expected currLayer maybeNext =
    case maybeNext of
      Nothing ->  -- We were given the last layer
        let lastOutputs = fmap neuronStateOutput currLayer
        in 2 *^ lastOutputs ^-^ expected

      Just (nextLayer, nextDeltas) ->
        let preacts = fmap neuronStatePreact currLayer -- z vector
            nextLayerWeightMat = fmap (neuronWeights . neuronStateNeuron)
                                      nextLayer
        in
          hadamardVec (transpose' nextLayerWeightMat !* nextDeltas)
                      (fmap (\neuronState -> neuronActDeriv (neuronStateNeuron neuronState)
                                                            (neuronStatePreact neuronState))
                            currLayer)

-- From: http://hackage.haskell.org/package/linear-1.20.9/docs/src/Linear.Vector.html#outer
outerInlined :: (Functor f, Functor g, Num a) => f a -> g a -> f (g a)
outerInlined a b = fmap (\x->fmap (*x) b) a
{-# INLINE outerInlined #-}

