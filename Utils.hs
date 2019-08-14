module Utils where

import           Data.Traversable
import           Data.Foldable

-- | This function is from:
-- https://wiki.haskell.org/Foldable_and_Traversable#Generalising_zipWith
zipWithTF :: (Traversable t, Foldable f) => (a -> b -> c) -> t a -> f b -> t c
zipWithTF g t f = snd (mapAccumL map_one (toList f) t)
  where map_one (x:xs) y = (xs, g y x)

zipTF :: (Traversable t, Foldable f) => t a -> f b -> t (a, b)
zipTF = zipWithTF (,)

