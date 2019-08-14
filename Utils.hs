{-# LANGUAGE ScopedTypeVariables #-}

module Utils where

import           Data.Traversable
import           Data.Foldable

import           GHC.Stack

-- | This function is from:
-- https://wiki.haskell.org/Foldable_and_Traversable#Generalising_zipWith
zipWithTF :: forall t f a b c. HasCallStack => (Traversable t, Foldable f) => (a -> b -> c) -> t a -> f b -> t c
zipWithTF g t f = snd (mapAccumL map_one (toList f) t)
  where
    map_one :: HasCallStack => [b] -> a -> ([b], c)
    map_one (x:xs) y = (xs, g y x)
    map_one _ _ = error "map_one"

zipTF :: HasCallStack => (Traversable t, Foldable f) => t a -> f b -> t (a, b)
zipTF = zipWithTF (,)

