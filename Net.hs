{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE LiberalTypeSynonyms #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Net where

import           Linear.Metric
import           Linear.Vector
import           Linear.Matrix hiding (trace)

import           Data.Functor.Compose

import           Data.Foldable
import           Data.List (mapAccumL)

import           Numeric.AD.Mode.Reverse

import           Data.Void

import           Control.Lens.Indexed
import           Control.Lens.At
import           Control.Lens ((^?))

import           Data.List (genericLength)

import           Dense
import           Backprop
import           Utils

-- import Debug.Trace

type Net      f a = f (Dense f a)
type NetState f a = f (f (NeuronState f a))

initNet :: (Traversable f, Monad m) =>
  (forall x. Int -> m x -> m (f x)) -> (forall x. [x] -> f x) -> Int -> f Int -> (a -> a) -> m a -> m a -> m (Net f a)
initNet replicateM' fromList numInputs0 sizes0 activationFn genWeight genBias =
  sequence (fromList (go numInputs0 (toList sizes0)))
  where
    go _         []           = []
    go numInputs (size:sizes) =
      initDense replicateM' numInputs size activationFn genWeight genBias
        : go size sizes

computeNetState :: (Metric f, Floating a) =>
  Net f a -> f a -> NetState f a
computeNetState net inputs = fmap (`denseOutput` inputs) net

netStateOutputs :: LayerCtx f a => NetState f a -> f a
netStateOutputs netState =
  let Just lastLayer = netState ^? ix (length netState-1)
  in
  fmap neuronStateOutput lastLayer

train :: forall f a. LayerCtx f a =>
  Int -> a -> DiffFn -> Net f a -> f (f (f a, f a)) -> Net f a
train nPasses stepSize sigma net0 minibatches =
    go nPasses net0
  where
    go 0       net = net
    go nPasses net =
      go (nPasses-1) (onePass net minibatches)

    onePass :: Net f a -> f (f a, f a) -> Net f a
    onePass net [] = net
    onePass net (currMinibatch:restMinibatches) =
      onePass (backprop stepSize sigma currMinibatch (computeNetState net)) restMinibatches
    

-- train :: LayerCtx f a =>
--   Int -> a -> DiffFn -> Net f a -> [(f a, f a)] -> Net f a
-- train 0     _        _     net _            = net
-- train iters stepSize sigma net trainingData =
--   let net' = trainOnce stepSize sigma net trainingData
--   in
--   train (iters-1) stepSize sigma net' trainingData

-- trainOnce :: LayerCtx f a =>
--   a -> DiffFn -> Net f a -> [(f a, f a)] -> Net f a
-- trainOnce stepSize sigma net [] = net
-- trainOnce stepSize sigma net ((currInput, currExpected):restTraining) =
--   let net' = backprop stepSize sigma currInput currExpected (computeNetState net currInput)
--   in
--   trainOnce stepSize sigma net' restTraining


-- | Percent accurate of identically correct results
netTestAccuracy :: (LayerCtx f a, Eq (f a)) =>
  (f a -> f a) -> DiffFn -> Net f a -> f (f a) -> f (f a) -> Double
netTestAccuracy classify sigma net testInputs testExpecteds =
  foldl' check 0 (zipTF netOutputs testExpecteds) / numTests
  where
    netOutputs = fmap (netStateOutputs . computeNetState net) testInputs

    numTests = fromIntegral $ length testInputs

    check r (xs, ys)
      | all withinTol (zipWithTF (-) (classify xs) ys) = r + 1
      | otherwise = r

    withinTol x = x <= 1e-6

