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

import           Control.Lens.At
import           Control.Lens.Indexed

import           Linear.Vector
import           Linear.Metric
import           Linear.Trace
import           Linear.Matrix
import           Data.Distributive

import           Data.Foldable
import           Data.Traversable

import           Utils

type DiffFn = forall a. (Floating a, Ord a, Eq a) => a -> a

data Neuron f a =
  Neuron
    { neuronWeights :: f a
    , neuronBias    :: a
    , neuronActFn   :: a -> a  -- | Aciviation function
    }

type Dense f a = f (Neuron f a)

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
  let neurons = zipWithTF mkNeuron weights biases
  return neurons
  where
    mkNeuron ws bs = Neuron ws bs activationFn

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
densePreactivated denseNeurons inputs =
  fmap (\n@Neuron{neuronWeights, neuronBias} ->
          (n, inputs `dot` neuronWeights + neuronBias))
       denseNeurons

denseOutput :: (Metric f, Floating a) => Dense f a -> f a -> f (NeuronState f a)
denseOutput d =
  fmap go . densePreactivated d
  where
    go (neuron@Neuron{neuronActFn}, preact) =
      NeuronState
        { neuronStateNeuron = neuron
        , neuronStatePreact = preact
        , neuronStateOutput = neuronActFn preact
        }

type LayerCtx f a = (Distributive f, Trace f, Index (f (f (NeuronState f a))) ~ Int, Index (f a) ~ Int,
     Index (f (NeuronState f a)) ~ Int,
     Ixed (f (NeuronState f a)),
     IxValue (f (NeuronState f a)) ~ NeuronState f a,
     IxValue (f (f (NeuronState f a))) ~ f (NeuronState f a),
     IxValue (f a) ~ a,
     Metric f, TraversableWithIndex Int f, Ixed (f a), Ixed (f (f (NeuronState f a))),
     Floating a, Ord a)

