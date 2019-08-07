module Parser where

import qualified Data.ByteString as BS
import           Data.ByteString (ByteString)

import qualified Data.Vector as V
import           Data.Vector (Vector)

import           Data.Word (Word8, Word32)
import           Data.Bits

import           Data.List.Split

import           Control.Monad

type Matrix a = Vector (Vector a)

parseVector :: ByteString -> Vector Word8
parseVector =
  V.fromList .
  drop (4*2) .    -- Drop the magic number and the number of elements (two
                  -- 4-byte integers)
  BS.unpack

parseImages :: Floating a => ByteString -> [Vector a]
parseImages = map (join . fmap (fmap scale)) . parseMatrix
  where
    scale = (/255) . fromIntegral

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

