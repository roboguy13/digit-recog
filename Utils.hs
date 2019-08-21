{-# LANGUAGE ScopedTypeVariables #-}

module Utils where

import           Data.Traversable
import           Data.Foldable

import           GHC.Stack

import           Linear.Vector

-- | This function is from:
-- https://wiki.haskell.org/Foldable_and_Traversable#Generalising_zipWith
zipWithTF :: forall t f a b c. -- HasCallStack =>
  (Traversable t, Foldable f) => (a -> b -> c) -> t a -> f b -> t c
zipWithTF g t f = snd (mapAccumL map_one (toList f) t)
  where
    map_one :: -- HasCallStack =>
      [b] -> a -> ([b], c)
    map_one (x:xs) y = (xs, g y x)
    map_one _ _ = error "map_one"
    {-# INLINE map_one #-}
{-# INLINE zipWithTF #-}

zipTF :: -- HasCallStack =>
  (Traversable t, Foldable f) => t a -> f b -> t (a, b)
zipTF = zipWithTF (,)
{-# INLINE zipTF #-}

shape1 :: Foldable f => f a -> Int
shape1 = length

shape2 :: Foldable f => f (f a) -> (Int, Int)
shape2 xs = (shape1 xs, length (head (toList xs)))

shape3 :: Foldable f => f (f (f a)) -> (Int, Int, Int)
shape3 xs =
  let (a, b) = shape2 xs
  in (a, b, length (head (toList (head (toList xs)))))

raggedShape2 :: Foldable f => f (f a) -> (Int, [Int])
raggedShape2 xs =
  (shape1 xs, map length (toList xs))

-- | Hadamard product

hadamardVec :: (Additive f, Num a) => f a -> f a -> f a
hadamardVec a b = liftI2 (*) a b

-- class Hadamard f where
--   hadamardVec :: f a -> f a -> f a

-- Some general, but potentially slow, implementations:
-- hadamardMatGeneral :: (Trace f, Additive f, Num a) => f (f a) -> f (f a) -> f (f a)
-- hadamardMatGeneral a b = fmap diagonal $ liftI2 outer a b
-- hadamardVecGeneral :: (Foldable f, Trace f, Additive f, Num a) => f a -> f a -> f a
-- hadamardVecGeneral a b = diagonal $ outer a b

