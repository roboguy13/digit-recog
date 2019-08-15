{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}

-- {-# OPTIONS_GHC -Wall #-}

module Backprop (backprop) where

import           Prelude hiding (unzip)

import           Control.Lens.At
import           Control.Lens.Indexed
import           Control.Lens ((^?))

import           Data.Traversable

import           Linear.Vector
import           Linear.Trace
import           Linear.Matrix

import           Numeric.AD.Mode.Reverse

import           Dense
import           Utils


import Debug.Trace

backprop :: forall f a.  (LayerCtx f a) =>
  a -> DiffFn -> f a -> f a -> f (f (NeuronState f a)) -> f (Dense f a)
backprop stepSize sigma inputs expected layers =
  zipWithTF processOneLayer cwGrads layersAndDeltas
  where

    processOneLayer ::
      f (f a) -> (f (NeuronState f a), f a) ->
      f (Neuron f a)
    processOneLayer grads (neuronStates, currDeltas) =
      traceShow (shape2 grads, shape1 neuronStates, shape1 currDeltas) $
      -- traceShow (shape2 grads, shape1 neuronStates, shape1 currDeltas, shape2 deltas, shape3 cwGrads) $
      zipWithTF processNeuron grads (zipTF neuronStates currDeltas)

    processNeuron :: f a -> (NeuronState f a, a) -> (Neuron f a)
    processNeuron grads (NeuronState{neuronStateNeuron}, delta) =
      Neuron
        { neuronWeights =
            neuronWeights neuronStateNeuron ^-^ (stepSize *^ grads)
        , neuronBias    =
            neuronBias neuronStateNeuron - (stepSize * delta)
        , neuronActFn = neuronActFn neuronStateNeuron
        }

    layersAndDeltas = zipTF layers deltas

    -- | Indexed by layer, then by incoming neuron, then by outgoing neuron
    -- (?)
    cwGrads :: f (f (f a))
    cwGrads = imap findCWGrad deltas

    deltas :: f (f a)
    deltas = snd $ mapAccumL findDeltas Nothing layers

    findDeltas maybeNext currLayer =
      let currDeltas = computeDeltas sigma expected currLayer maybeNext
      in (Just (currLayer, currDeltas), currDeltas)

    findCWGrad currIx currDeltas =
      costWeightGrad inputs currDeltas
                     (layers ^? ix (currIx-1))

-- | In the result: The outermost "container" is indexed by incoming neuron
-- index (j) and the innermost "container" is indexed by the outgoing neuron
-- index (k)      (?)
costWeightGrad :: forall f a. LayerCtx f a =>
  f a -> f a ->
  Maybe (f (NeuronState f a)) ->
  f (f a)
costWeightGrad inputs currDeltas maybePrevLayer =
  outer currDeltas prevA
  where
    prevA =
      case maybePrevLayer of
        Nothing        -> inputs
        Just prevLayer -> fmap neuronStateOutput prevLayer


-- | `maybeNext` is the next layer together with the deltas from the next
-- layer. `expected` is the expected outputs.
--
-- The result is the deltas for the given layer.
computeDeltas :: forall f a. LayerCtx f a =>
  DiffFn -> f a ->
  f (NeuronState f a) -> Maybe (f (NeuronState f a), f a) -> f a
computeDeltas sigma expected currLayer maybeNext =
    case maybeNext of
      Nothing ->  -- We were given the last layer
        let lastOutputs = fmap neuronStateOutput currLayer
        in 2 *^ lastOutputs ^-^ expected

      Just (nextLayer, nextDeltas) ->
        let preacts = fmap neuronStatePreact currLayer -- z vector
            nextLayerWeightMat = fmap (neuronWeights . neuronStateNeuron)
                                      nextLayer
        in hadamardVec (transpose' nextLayerWeightMat !* nextDeltas)
                       (fmap sigmaDeriv preacts)
  where
    sigmaDeriv     = diff sigma

-- | Hadamard product
hadamardMat :: (Trace f, Additive f, Num a) => f (f a) -> f (f a) -> f (f a)
hadamardMat a b = fmap diagonal $ liftI2 outer a b

hadamardVec :: (Trace f, Additive f, Num a) => f a -> f a -> f a
hadamardVec a b = diagonal $ outer a b


