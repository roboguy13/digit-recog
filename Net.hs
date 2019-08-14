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

import           Dense
import           Backprop

type Net f a = f (Dense f a)
type NetState f a = f (f (NeuronState f a))

initNet :: (Traversable f, Monad m) =>
  (forall x. Int -> m x -> m (f x)) -> f Int -> (a -> a) -> m a -> m a -> m (Net f a)
initNet replicateM' sizes activationFn genWeight genBias =
  mapM (\size -> initDense replicateM' size activationFn genWeight genBias)
       sizes

computeNetState :: (Metric f, Floating a) =>
  Net f a -> f a -> NetState f a
computeNetState net inputs = fmap (`denseOutput` inputs) net

