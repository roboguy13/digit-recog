{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE Strict #-}

-- {-# OPTIONS_GHC -Wall #-}

module Dense where

import           Control.Lens.At
import           Control.Lens.Indexed

import           Linear.Vector
import           Linear.Metric
import           Linear.Trace
import           Linear.Matrix hiding (transpose)

import           Numeric.AD.Mode.Reverse

import           Data.List

import           Data.Vector (Vector)
import qualified Data.Vector as V

import           Data.Foldable
import           Data.Traversable

import           Utils

type DiffFn = forall a. (Floating a, Ord a, Eq a) => a -> a

data Neuron f a =
  Neuron
    { neuronWeights :: f a
    , neuronBias    :: a
    , neuronActFn   :: DiffFn  -- | Activation function
    , neuronActDeriv :: DiffFn
    }

-- | For testing and debugging purposes
instance (Show a, Show (f a)) => Show (Neuron f a) where
  show (Neuron ws bs _ _) = "Neuron (" ++ show ws ++ ", " ++ show bs ++ ")"

type Dense f a = f (Neuron f a)

data NeuronState f a =
  NeuronState
    { neuronStateNeuron :: Neuron f a
    , neuronStatePreact :: a -- | "Preactivated" result (often called z)
    , neuronStateOutput :: a -- | Output
    }
    deriving Show

initDense :: (Traversable f, Monad m) =>
  (forall x. Int -> m x -> m (f x)) -> Int -> Int -> DiffFn -> (Int -> m a) -> (Int -> m a) -> m (Dense f a)
initDense replicateM' numInputs size activationFn genWeight genBias = do
  weights <- replicateM' size (replicateM' numInputs (genWeight size))
  biases  <- replicateM' size (genBias size)
  let neurons = zipWithTF mkNeuron weights biases
  return neurons
  where
    mkNeuron ws bs = Neuron ws bs activationFn (diff activationFn)

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

-- | Takes an result from 'densePreactivated' as its argument
denseActivated :: (Metric f, Floating a, Ord a) => f (Neuron f a, a) -> f (NeuronState f a)
denseActivated = fmap go
  where
    go (neuron@Neuron{neuronActFn}, preact) =
      NeuronState
        { neuronStateNeuron = neuron
        , neuronStatePreact = preact
        , neuronStateOutput = neuronActFn preact
        }

denseOutput :: (Metric f, Floating a, Ord a) => Dense f a -> f a -> f (NeuronState f a)
denseOutput d = denseActivated . densePreactivated d

type LayerCtx f a = (Transpose f, Trace f, Index (f (f (NeuronState f a))) ~ Int, Index (f a) ~ Int,
     Index (f (f (Neuron f a))) ~ Int,
     IxValue (f (f (Neuron f a))) ~ f (Neuron f a),
     Index (f (NeuronState f a)) ~ Int,
     Ixed (f (f (Neuron f a))),
     Ixed (f (NeuronState f a)),
     IxValue (f (NeuronState f a)) ~ NeuronState f a,
     IxValue (f (f (NeuronState f a))) ~ f (NeuronState f a),
     IxValue (f a) ~ a,
     Metric f, TraversableWithIndex Int f, Ixed (f a), Ixed (f (f (NeuronState f a))),
     Floating a, Ord a)

class Transpose f where
  transpose' :: f (f a) -> f (f a)
  asColumn   :: f a -> f (f a)
  asRow      :: f a -> f (f a)

instance Transpose [] where
  transpose' = transpose
  asColumn = map (:[])
  asRow    = (:[])

instance Transpose Vector where
  transpose' m = -- TODO: Write a faster implementation
    V.fromList (fmap V.fromList (transpose (V.toList (fmap V.toList m))))

  asColumn = fmap V.singleton
  asRow    = V.singleton

