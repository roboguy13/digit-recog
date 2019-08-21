{-# LANGUAGE OverloadedLists #-}

module Main where

import qualified Data.ByteString as BS
import qualified Data.Vector as V
import           Data.Vector (Vector)

import           Control.Monad (replicateM)
import           Control.Arrow ((&&&))
-- import           System.Random

import           Linear.Trace

import           Data.List.Split (chunksOf)
import           Data.List (nub)

import           Linear.Vector


import           Net
import           Dense --(softplus, sigmoid, reLU)
import           Backprop (Net, computeNetState)
import           Parser
import           Utils

import           System.Random.MWC
import           Statistics.Distribution
import           Statistics.Distribution.Normal

imageSize :: Int
imageSize = 28*28

stepSize :: Double
stepSize = 0.6
-- stepSize = 0.15

genWeight :: GenIO -> Int -> IO Double
genWeight gen numWeights = genContVar (normalDistr 0 (1/sqrt (fromIntegral numWeights))) gen

genBias :: GenIO -> Int -> IO Double
genBias gen _numWeights = genContVar (normalDistr 0 1) gen

actFn :: Floating a => a -> a
actFn = sigmoid

boolToDouble :: Bool -> Double
boolToDouble False = 0
boolToDouble True  = 1

instance Trace Vector where
  diagonal m =
    V.fromList $ map (\i -> (m V.! i) V.! i) [0..size-1]
    where
      size = V.length m

classify :: Vector Double -> Vector Double
classify v =
  let idx = V.maxIndex v
  in fmap boolToDouble (oneHotEncode (fromIntegral idx))

main :: IO ()
main = do
  -- setStdGen (mkStdGen 200)
  -- setStdGen (mkStdGen 203)
  genIO <- createSystemRandom

  trainLabels0 <- parseLabels <$> BS.readFile "images/train/train-labels-idx1-ubyte"
  trainImages <- parseImages <$> BS.readFile "images/train/train-images-idx3-ubyte"

  testLabels0 <- parseLabels <$> BS.readFile "images/test/t10k-labels-idx1-ubyte"
  testImages <- parseImages <$> BS.readFile "images/test/t10k-images-idx3-ubyte"


  let trainLabels = fmap (fmap boolToDouble) trainLabels0
      testLabels  = fmap (fmap boolToDouble) testLabels0

  let testSampleSize   = 1000
      sampleTestImages = V.take testSampleSize testImages
      sampleTestLabels = V.take testSampleSize testLabels

  let trainLabelsAndImages
        = zip (V.toList trainImages) (V.toList trainLabels)

  initialNet <-
    initNet V.replicateM V.fromList imageSize [16, 16, 10] actFn (genWeight genIO) (genBias genIO)
      :: IO (Net Vector Double)

  let trainedNet =
        train 50 stepSize initialNet (map V.fromList $ chunksOf 10 (take 700 trainLabelsAndImages))
        -- train 500 stepSize actFn initialNet [V.fromList $ take 3 trainLabelsAndImages]
        -- train 10 stepSize actFn initialNet (take 2 (map V.fromList $ chunksOf 200 trainLabelsAndImages))

  putStr "Training accuracy: "
  putStr (show (netTestAccuracy classify trainedNet (V.take 1000 trainImages) (V.take 1000 trainLabels)*100))
  putStrLn "%"

  putStr "Test accuracy: "
  putStr (show (netTestAccuracy classify trainedNet testImages testLabels*100))
  putStrLn "%"

