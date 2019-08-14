{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE LiberalTypeSynonyms #-}
module Net where

import           Linear.Metric
import           Linear.Vector
import           Linear.Matrix

import           Data.Functor.Compose

import           Data.Foldable
import           Data.List (mapAccumL)

import           Numeric.AD.Mode.Reverse

import           Data.Void

import           Control.Lens.Indexed
import           Control.Lens.At
import           Control.Lens ((^?))

import           Dense
import           Backprop

type Net      f a = f (Dense f a)
type NetState f a = f (f (NeuronState f a))

initNet :: (Traversable f, Monad m) =>
  (forall x. Int -> m x -> m (f x)) -> f Int -> (a -> a) -> m a -> m a -> m (Net f a)
initNet replicateM' sizes activationFn genWeight genBias =
  mapM (\size -> initDense replicateM' size activationFn genWeight genBias)
       sizes

computeNetState :: (Metric f, Floating a) =>
  Net f a -> f a -> NetState f a
computeNetState net inputs = fmap (`denseOutput` inputs) net

netStateOutputs :: LayerCtx f a => NetState f a -> f a
netStateOutputs netState =
  let Just lastLayer = netState ^? ix (length netState-1)
  in
  fmap neuronStateOutput lastLayer

backpropNet :: LayerCtx f a =>
  a -> DiffFn -> f a -> f a -> NetState f a -> Net f a
backpropNet = backprop

-- | Train a 'Net' on a list of (input, expected output) pairs, one at a time
train :: LayerCtx f a =>
  a -> DiffFn -> Net f a -> [(f a, f a)] -> Net f a
train _        _     net []          = net
train stepSize sigma net ((currInput, currExpected):restTraining) =
  train stepSize sigma net' restTraining
  where
    net' = backpropNet stepSize sigma currInput currExpected (computeNetState net currInput)

