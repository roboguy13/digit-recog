{-# LANGUAGE Strict #-}

module Parser (parseImages, parseLabels, oneHotEncode, Matrix) where

import qualified Data.ByteString as BS
import           Data.ByteString (ByteString)

import qualified Data.Vector as V
import           Data.Vector (Vector)

import           Data.Word (Word8, Word32)
import           Data.Bits

import           Data.List.Split

import           Control.Monad

type Matrix a = Vector (Vector a)

parseImages :: Floating a => ByteString -> Matrix a
parseImages = V.fromList . map (join . fmap (fmap scale)) . parseMatrix
  where
    scale = (/255) . fromIntegral


-- | Using a one-hot encoding in the result
parseLabels :: ByteString -> Matrix Bool
parseLabels bytes =
  let labelData = drop (4*2) (BS.unpack bytes)
  in V.fromList $ map oneHotEncode labelData

oneHotEncode :: Word8 -> Vector Bool
oneHotEncode n
  | n < 0 || n > 9 = error ("Invalid label: " ++ show n)
  | otherwise =
      V.fromList theList
  where
    theList =
      replicate' n False ++ [True] ++ replicate' (9-n) False

    replicate' = replicate . fromIntegral

parseMatrix :: ByteString -> [Matrix Word8]
parseMatrix bytes =
  let (nCols, restBytes) = splitAt 4 $ drop (4*2) (BS.unpack bytes)
      (nRows, imageData) = splitAt 4 restBytes
      rows               = chunksOf (fromIntegral (bytesTo32 nCols)) imageData
      images             = chunksOf (fromIntegral (bytesTo32 nRows)) rows
  in map (V.fromList . map V.fromList) images

-- | NOTE: Requires exactly 4 bytes in the given list.
bytesTo32 :: [Word8] -> Word32
bytesTo32 [a, b, c, d] =
  fromIntegral d + (fromIntegral c `shiftL` 8)
                 + (fromIntegral b `shiftL` (8*2))
                 + (fromIntegral a `shiftL` (8*3))
bytesTo32 _ =
  error "bytesTo32: Wrong number of bytes. Requires exactly 4 bytes."

