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
import           Dense (softplus, sigmoid, reLU)
import           Parser
import           Utils

imageSize :: Int
imageSize = 28*28

stepSize :: Double
-- stepSize = 0.6
stepSize = 0.6

between0and1 :: IO Double
between0and1 = randomRIO (0,1)

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
  setStdGen (mkStdGen 200)

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
    initNet V.replicateM V.fromList imageSize [16, 16, 10] actFn between0and1 between0and1
      :: IO (Net Vector Double)

  let trainedNet = train 1 stepSize actFn initialNet [V.fromList $ take 6000 trainLabelsAndImages]

  -- print trainedNet

  putStr "Test accuracy: "
  putStr (show (netTestAccuracy classify actFn trainedNet sampleTestImages sampleTestLabels*100))
  putStrLn "%"

