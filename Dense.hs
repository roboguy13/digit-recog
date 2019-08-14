{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- {-# OPTIONS_GHC -Wall #-}

module Dense where

import           Linear.Vector
import           Linear.Metric
import           Linear.Trace
import           Linear.Matrix
import           Data.Distributive

import           Data.Foldable
import           Data.Traversable

import           Numeric.AD.Mode.Reverse

import           Control.Lens.At
import           Control.Lens.Indexed
import           Control.Lens ((^?))

-- | Hadamard product
hadamardMat :: (Trace f, Additive f, Num a) => f (f a) -> f (f a) -> f (f a)
hadamardMat a b = fmap diagonal $ liftI2 outer a b

hadamardVec :: (Trace f, Additive f, Num a) => f a -> f a -> f a
hadamardVec a b = diagonal $ outer a b

type DiffFn = forall a. (Floating a, Ord a, Eq a) => a -> a

data Neuron f a =
  Neuron
    { neuronWeights :: f a
    , neuronBias    :: a
    }

data Dense f a =
  Dense
    { denseNeurons :: f (Neuron f a)
    , activationFn :: a -> a
    }

data NeuronState f a =
  NeuronState
    { neuronStateNeuron :: Neuron f a
    , neuronStatePreact :: a -- | "Preactivated" result (often called z)
    , neuronStateOutput :: a -- | Output
    }

initDense :: (Traversable f, Monad m) =>
  (forall x. Int -> m x -> m (f x)) -> Int -> (a -> a) -> m a -> m a -> m (Dense f a)
initDense replicateM' size activationFn genWeight genBias = do
  weights <- replicateM' size (replicateM' size genWeight)
  biases  <- replicateM' size genBias
  let neurons = zipWithTF Neuron weights biases
  return $ Dense { denseNeurons = neurons, activationFn = activationFn }

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))

reLU :: (Num a, Ord a) => a -> a
reLU = max 0

leakyReLU :: (Floating a, Ord a) => a -> a -> a
leakyReLU epsilon x
  | x > 0     = x
  | otherwise = epsilon*x

-- | A smooth approximation of ReLU
softplus :: Floating a => a -> a
softplus x = log (1 + exp x)

densePreactivated :: (Metric f, Floating a) => Dense f a -> f a -> f (Neuron f a, a)
densePreactivated Dense{denseNeurons} inputs =
  fmap (\n@Neuron{neuronWeights, neuronBias} ->
          (n, inputs `dot` neuronWeights + neuronBias))
       denseNeurons

denseOutput :: (Metric f, Floating a) => Dense f a -> f a -> f (NeuronState f a)
denseOutput d@Dense{activationFn} =
  fmap go . densePreactivated d
  where
    go (neuron, preact) =
      NeuronState
        { neuronStateNeuron = neuron
        , neuronStatePreact = preact
        , neuronStateOutput = activationFn preact
        }

backprop :: forall f a.  LayerCtx f a =>
  a -> DiffFn -> f a -> f a -> f (f (NeuronState f a)) -> f (f (Neuron f a))
backprop stepSize sigma inputs expected layers =
  zipWithTF processOneLayer cwGrads layersAndDeltas
  where

    processOneLayer ::
      f (f a) -> (f (NeuronState f a), f a) ->
      f (Neuron f a)
    processOneLayer grads (neuronStates, currDeltas) =
      zipWithTF processNeuron grads (zipTF neuronStates currDeltas)

    processNeuron :: f a -> (NeuronState f a, a) -> Neuron f a
    processNeuron grads (NeuronState{neuronStateNeuron}, delta) =
      Neuron
        { neuronWeights =
            neuronWeights neuronStateNeuron ^-^ (stepSize *^ grads)
        , neuronBias    =
            neuronBias neuronStateNeuron - (stepSize * delta)
        }

    layersAndDeltas = zipTF layers deltas

    -- | Indexed by layer, then by incoming neuron, then by outgoing neuron
    -- (?)
    cwGrads :: f (f (f a))
    cwGrads = imap findCWGrad deltas

    deltas :: f (f a)
    deltas = snd $ mapAccumR findDeltas Nothing layers

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
        in hadamardVec (transpose nextLayerWeightMat !* nextDeltas)
                       (fmap sigmaDeriv preacts)
  where
    sigmaDeriv     = diff sigma

type LayerCtx f a = (Distributive f, Trace f, Index (f (f (NeuronState f a))) ~ Int, Index (f a) ~ Int,
     Index (f (NeuronState f a)) ~ Int,
     Ixed (f (NeuronState f a)),
     IxValue (f (NeuronState f a)) ~ NeuronState f a,
     IxValue (f (f (NeuronState f a))) ~ f (NeuronState f a),
     IxValue (f a) ~ a,
     Metric f, TraversableWithIndex Int f, Ixed (f a), Ixed (f (f (NeuronState f a))),
     Floating a, Ord a)

-- | This function is from:
-- https://wiki.haskell.org/Foldable_and_Traversable#Generalising_zipWith
zipWithTF :: (Traversable t, Foldable f) => (a -> b -> c) -> t a -> f b -> t c
zipWithTF g t f = snd (mapAccumL map_one (toList f) t)
  where map_one (x:xs) y = (xs, g y x)

zipTF :: (Traversable t, Foldable f) => t a -> f b -> t (a, b)
zipTF = zipWithTF (,)

