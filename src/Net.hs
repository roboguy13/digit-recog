{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE LiberalTypeSynonyms #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE Strict #-}

module Net where

import           Linear.Metric
import           Linear.Vector
import           Linear.Matrix hiding (trace)

import           Data.Functor.Compose

import           Data.Foldable
import           Data.List (mapAccumL, mapAccumR)

import           Numeric.AD.Mode.Reverse

import           Data.Void

import           Control.Lens.Indexed
import           Control.Lens.At
import           Control.Lens ((^?))

import           Data.List (genericLength)

import           Dense
import           Backprop
import           Utils

import Debug.Trace

initNet :: (Traversable f, Monad m) =>
  (forall x. Int -> m x -> m (f x)) -> (forall x. [x] -> f x) -> Int -> f Int ->
  DiffFn -> (Int -> m a) -> (Int -> m a) -> m (Net f a)
initNet replicateM' fromList numInputs0 sizes0 activationFn genWeight genBias =
  sequence (fromList (go numInputs0 (toList sizes0)))
  where
    go _         []           = []
    go numInputs (size:sizes) =
      initDense replicateM' numInputs size activationFn genWeight genBias
        : go size sizes


netStateOutputs :: LayerCtx f a => NetState f a -> f a
netStateOutputs netState =
  let Just lastLayer = netState ^? ix (length netState-1)
  in
  fmap neuronStateOutput lastLayer

train :: forall f a. (Show a, LayerCtx f a) =>
  Int -> a -> Net f a -> [f (f a, f a)] -> Net f a
train nPasses stepSize net0 minibatches =
    go nPasses net0
  where
    go 0 net = net
    go n net = go (n-1) (onePass net)

    onePass :: Net f a -> Net f a
    onePass net =
      foldl' (\currNet minibatch -> backprop (length minibatch) stepSize minibatch currNet)
             net
             minibatches

-- | Percent accurate of identically correct results
netTestAccuracy :: (Show (f a), Show (f (f a)), LayerCtx f a, Eq (f a)) =>
  (f a -> f a) -> Net f a -> f (f a) -> f (f a) -> Double
netTestAccuracy classify net testInputs testExpecteds =
  foldl' check 0 (zipTF netOutputs testExpecteds) / numTests
  where
    netOutputs = fmap (netStateOutputs . computeNetState net) testInputs

    numTests = fromIntegral $ length testInputs

    check r (xs, ys)
      | all withinTol (zipWithTF (-) (classify xs) ys) = r + 1
      | otherwise = r

    withinTol x = x <= 1e-6

