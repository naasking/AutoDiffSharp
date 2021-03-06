﻿using System;
using System.Linq;
using Xunit;

namespace AutoDiffSharp.Tests
{
    public static class Tests
    {
        [Fact]
        public static void BasicTests()
        {
            Assert.Equal(1, new Dual(1).Magnitude);
            Assert.Equal(3, (new Dual(1) + new Dual(2)).Magnitude);
            Assert.Equal(new Dual(1), new Dual(1));
            Assert.NotEqual(new Dual(2), new Dual(1));
            var x = new Dual(2);
            Assert.Equal(4, (x + x).Magnitude);
            Assert.Equal(0, (x + x).Derivative);
            Assert.Equal(0, (x - x).Magnitude);
            Assert.Equal(0, (-x + x).Magnitude);
            Assert.Equal(6, (x * 3).Magnitude);
            Assert.Equal(6, (3 * x).Magnitude);
            Assert.Equal(Math.Sin(x.Magnitude), x.Sin().Magnitude);
            Assert.Equal(Math.Cos(x.Magnitude), x.Cos().Magnitude);
            Assert.Equal(4, (x ^ 2).Magnitude);
            Assert.Equal(4, (x * x).Magnitude);
            Assert.Equal(x, x);
            Assert.Equal(0, x.CompareTo(x));
            //Assert.Equal(default, x - x);
            Assert.Equal(Math.Cos(3), Calculus.DerivativeAt(3, (Dual x) => x.Cos()).Magnitude);
            Assert.Equal(-Math.Sin(3), Calculus.DerivativeAt(3, (Dual x) => x.Cos()).Derivative);
            Assert.Equal(Math.Sin(3), Calculus.DerivativeAt(3, (Dual x) => x.Sin()).Magnitude);
            Assert.Equal(Math.Cos(3), Calculus.DerivativeAt(3, (Dual x) => x.Sin()).Derivative);
        }

        [Fact]
        public static void TestPow()
        {
            var x = new Dual(25);
            Assert.Equal(x * x, x ^ 2);
        }

        [Fact]
        public static void TestPoly2()
        {
            var z = Calculus.DerivativeAt(5, 2, SimplePoly2);
            Assert.Equal(59, z.Magnitude);
            Assert.Equal(30, z.Derivative(0));
            Assert.Equal(-24, z.Derivative(1));
        }

        static Codual SimplePoly2(Codual x, Codual y) =>
            (x ^ 2) * 3 - (y ^ 3) * 2;

        [Fact]
        public static void TestPoly3()
        {
            var z = Calculus.DerivativeAt(1, 0.5, 2, SimplePoly3);
            Assert.Equal(4.5, z.Magnitude);
            Assert.Equal(6, z.Derivative(0));
        }

        static Codual SimplePoly3(Codual a, Codual b, Codual c) =>
            c * (a + b).Pow(2);

        [Theory]
        [InlineData(5, 2, 4)]
        [InlineData(1, 0, 0)]
        [InlineData(2, 1, 2)]
        [InlineData(17, 4, 8)]
        [InlineData(82, 9, 18)]
        [InlineData(2, -1, -2)]
        public static void TestFunc(double y, double x, double dx)
        {
            var dy = Calculus.DerivativeAt(x, Func);
            Assert.Equal(y, dy.Magnitude);
            Assert.Equal(dx, dy.Derivative);
        }

        static Dual Func(Dual x) =>
            (x ^ 2) + 1;

        [Theory]
        [InlineData(0, 0, 2, 0, 0)]
        [InlineData(1, 0, 2, 1, 3)]
        [InlineData(2, 1, 2, 0, 0)]
        [InlineData(3, 1, 2, 1, 3)]
        [InlineData(12, 2, 2, 2, 12)]
        public static void TestFunc2(double f, double y, double dy, double x, double dx)
        {
            var df = Calculus.DerivativeAt(x, y, Func2);
            Assert.Equal(f, df.Magnitude);
            Assert.Equal(dx, df.Derivative(0));
            Assert.Equal(dy, df.Derivative(1));
        }

        static Codual Func2(Codual x, Codual y) =>
            (x ^ 3) + y * 2;

        [Theory]
        [InlineData(0, 0, 0, 0, 0)]
        [InlineData(1, 0, 2, 1, 3)]
        [InlineData(0, 1, 0, 0, 2)]
        [InlineData(3, 1, 2, 1, 5)]
        [InlineData(16, 2, 4, 2, 16)]
        public static void TestFunc3(double f, double y, double dy, double x, double dx)
        {
            var df = Calculus.DerivativeAt(x, y, Func3);
            Assert.Equal(f, df.Magnitude);
            Assert.Equal(dx, df.Derivative(0));
            Assert.Equal(dy, df.Derivative(1));
        }

        static Codual Func3(Codual x, Codual y) =>
            (x ^ 3) + y * x * 2;

        [Theory]
        [InlineData(0, 0, 0)]
        [InlineData(3, 1, 11)]
        [InlineData(48, 2, 104)]
        [InlineData(-3, -1, 11)]
        public static void TestQuintile(double y, double x, double dx)
        {
            var dy = Calculus.DerivativeAt(x, Quintile);
            Assert.Equal(y, dy.Magnitude);
            Assert.Equal(dx, dy.Derivative);
        }

        static Dual Quintile(Dual x) =>
            (x ^ 5) + 2 * (x ^ 3);

        [Theory]
        [InlineData(0, 0, 0)]
        [InlineData(3, 1, 11)]
        [InlineData(48, 2, 104)]
        [InlineData(-3, -1, 11)]
        public static void TestQuintile2(double y, double x, double dx)
        {
            var dy = Calculus.DerivativeAt(x, Quintile2);
            Assert.Equal(y, dy.Magnitude);
            Assert.Equal(dx, dy.Derivative);
        }

        // Same as Quintile, just refactored to ensure outcome is the same
        static Dual Quintile2(Dual x) =>
            (x ^ 3) * ((x ^ 2) + 2);

        [Fact]
        public static void TestSample()
        {
            var y = Calculus.DerivativeAt(2, 3, Sample);
            Assert.Equal(5.947664043757056, y.Magnitude);
            Assert.Equal(1.0013704652454263, y.Derivative(1));
        }

        static Codual Sample(Codual x1, Codual x2) =>
            x1 * x2 - x2.SinDeg();

        [Fact()]
        public static void TestScaling()
        {
            var y = Calculus.DerivativeAt(2, 1, Scaling);
        }

        static Codual Scaling(Codual x0, Codual x1)
        {
            for (int i = 0; i < 10; ++i)
                x1 = x1 * 2 + x1 * x0;
                //x1 = (x1 * 2 / x1);
                //x1 = x1 * (x1 * 2 / 4 + 1);
            return x1;
        }
    }
}
