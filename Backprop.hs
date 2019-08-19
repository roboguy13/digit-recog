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
import           Linear.Trace hiding (trace)
import           Linear.Matrix hiding (trace)

import           Numeric.AD.Mode.Reverse

import           Dense
import           Utils


-- import Debug.Trace

backprop :: forall f a.  (LayerCtx f a) =>
  a -> DiffFn -> f (f a, f a) -> f (f (Neuron f a)) -> f (Dense f a)
backprop stepSize sigma minibatch layers =
  zipWithTF processOneLayer cwGrads layersAndDeltas
  where

    processOneLayer ::
      f (f (f a)) -> (f (Neuron f a), f (f a)) ->
      f (Neuron f a)
    processOneLayer grads (neuronStates, currDeltas) =
      zipWithTF processNeuron grads (zipTF neuronStates currDeltas)

    processNeuron :: f (f a) -> (Neuron f a, f a) -> (Neuron f a)
    processNeuron grads (neuron, delta) =
      Neuron
        { neuronWeights =
            foldr (^+^) zero $
              fmap (\currGrads -> 
                      neuronWeights neuron ^-^ (stepSize *^ currGrads))
                   grads
        , neuronBias    =
            sum $
              fmap (\currDelta ->
                      neuronBias neuron - (stepSize * currDelta))
                   delta
        , neuronActFn = neuronActFn neuron
        }

    layersAndDeltas :: f (f (Neuron f a), f (f a))
    layersAndDeltas = zipTF layers deltas

    -- | Indexed by mini-batch index, then by layer, then by incoming neuron, then by outgoing neuron
    -- (?)
    cwGrads :: f (f (f (f a)))
    cwGrads =
      zipWithTF
        (\currInputs currDeltas -> imap (findCWGrad currInputs) currDeltas)
        (fmap fst minibatch)
        deltas

    -- Indexed first by mini-batch index
    deltas :: f (f (f a))
    deltas =
      fmap (\(currInputs, currExpected) ->
                  snd $ mapAccumR (findDeltas currExpected) Nothing _)
           minibatch

    findDeltas :: f a
      -> Maybe (f (NeuronState f a), f a)
      -> f (NeuronState f a)
      -> (Maybe (f (NeuronState f a), f a), f a)
    findDeltas currExpected maybeNext currLayer =
      let currDeltas = computeDeltas sigma currExpected currLayer maybeNext
      in (Just (currLayer, currDeltas), currDeltas)

    findCWGrad :: f a -> Int -> f a -> f (f a)
    findCWGrad currInputs currIx currDeltas =
      costWeightGrad currInputs currDeltas
                     (layers ^? ix (currIx-1))

-- | In the result: The outermost "container" is indexed by incoming neuron
-- index (j) and the innermost "container" is indexed by the outgoing neuron
-- index (k)      (?)
costWeightGrad :: forall f a. LayerCtx f a =>
  f a -> f a ->
  Maybe (f (Neuron f a)) ->
  f (f a)
costWeightGrad inputs currDeltas maybePrevLayer =
  let result = outer currDeltas prevA
  in result
  where
    prevA =
      case maybePrevLayer of
        Nothing        -> inputs
        Just prevLayer -> fmap neuronStateOutput $ denseOutput prevLayer inputs
        -- Just prevLayer -> fmap neuronStateOutput prevLayer


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
        in
          hadamardVec (transpose' nextLayerWeightMat !* nextDeltas)
                      (fmap sigmaDeriv preacts)
  where
    sigmaDeriv     = diff sigma

-- | Hadamard product
hadamardMat :: (Trace f, Additive f, Num a) => f (f a) -> f (f a) -> f (f a)
hadamardMat a b = fmap diagonal $ liftI2 outer a b

hadamardVec :: (Foldable f, Trace f, Additive f, Num a) => f a -> f a -> f a
hadamardVec a b =
  let result = diagonal $ outer a b
  in
    result


