{-# LANGUAGE OverloadedLists #-}

module Main where

import qualified Data.ByteString as BS
import qualified Data.Vector as V
import           Data.Vector (Vector)

import           Control.Monad (replicateM)
import           Control.Arrow ((&&&))
import           System.Random

import           Linear.Trace

import           Net
import           Dense (softplus)
import           Parser
import           Utils

imageSize :: Int
imageSize = 28*28

between0and1 :: IO Double
between0and1 = randomRIO (0,1)

actFn :: Floating a => a -> a
actFn = softplus

boolToDouble :: Bool -> Double
boolToDouble False = 0
boolToDouble True  = 1

instance Trace Vector where
  diagonal m =
    V.fromList $ map (\i -> (m V.! i) V.! i) [0..size-1]
    where
      size = V.length m


main :: IO ()
main = do
  setStdGen (mkStdGen 27)

  trainLabels0 <- parseLabels <$> BS.readFile "images/train/train-labels-idx1-ubyte"
  trainImages <- parseImages <$> BS.readFile "images/train/train-images-idx3-ubyte"

  testLabels0 <- parseLabels <$> BS.readFile "images/test/t10k-labels-idx1-ubyte"
  testImages <- parseImages <$> BS.readFile "images/test/t10k-images-idx3-ubyte"


  let trainLabels = fmap (fmap boolToDouble) trainLabels0
      testLabels  = fmap (fmap boolToDouble) testLabels0

  let sampleTestImages = V.take 50 testImages
      sampleTestLabels = V.take 50 testLabels

  let trainLabelsAndImages
        = zip (V.toList trainImages) (V.toList trainLabels)

  initialNet <-
    initNet V.replicateM V.fromList imageSize [16, 16, 10] actFn between0and1 between0and1
      :: IO (Net Vector Double)

  print (shape2 trainLabels, shape2 trainImages)
  print (fmap shape1 (V.take 10 trainLabels))
  print (fmap shape1 (V.take 10 trainImages))

  let trainedNet = train 1 0.01 actFn initialNet (take 2 trainLabelsAndImages)

  print (length trainImages)
  print trainedNet

  -- print (netTestAccuracy 0.01 actFn trainedNet sampleTestImages sampleTestLabels)

