using System;
using System.Linq;
using Xunit;

namespace AutoDiffSharp.Tests
{
    public static class Tests
    {
        [Fact]
        public static void BasicTests()
        {
            Assert.Equal(1, new Number(1).Magnitude);
            Assert.Equal(3, (new Number(1) + new Number(2)).Magnitude);
            Assert.Equal(new Number(1), new Number(1));
            Assert.NotEqual(new Number(2), new Number(1));
            var x = new Number(2);
            Assert.Equal(4, (x + x).Magnitude);
            Assert.Equal(0, (x + x).Derivatives[0]);
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
            Assert.Equal(Math.Cos(3), Calculus.DifferentiateAt(3, x => x.Cos()).Magnitude);
            Assert.Equal(-Math.Sin(3), Calculus.DifferentiateAt(3, x => x.Cos()).Derivatives[0]);
            Assert.Equal(Math.Sin(3), Calculus.DifferentiateAt(3, x => x.Sin()).Magnitude);
            Assert.Equal(Math.Cos(3), Calculus.DifferentiateAt(3, x => x.Sin()).Derivatives[0]);
        }

        [Fact]
        public static void TestPow()
        {
            var x = new Number(25);
            Assert.Equal(x * x, x ^ 2);
        }

        [Fact]
        public static void TestPoly2()
        {
            var z = Calculus.DifferentiateAt(5, 2, SimplePoly2);
            Assert.Equal(59, z.Magnitude);
            Assert.Equal(30, z.Derivatives[0]);
            Assert.Equal(-24, z.Derivatives[1]);
        }

        static Number SimplePoly2(Number x, Number y) =>
            3 * (x ^ 2) - 2 * (y ^ 3);

        [Fact]
        public static void TestPoly3()
        {
            var z = Calculus.DifferentiateAt(1, 0.5, 2, SimplePoly3);
            Assert.Equal(4.5, z.Magnitude);
            Assert.Equal(6, z.Derivatives[0]);
        }

        static Number SimplePoly3(Number a, Number b, Number c) =>
            c * (a + b).Pow(2);

        [Theory]
        [InlineData(5, 2, 4)]
        [InlineData(1, 0, 0)]
        [InlineData(2, 1, 2)]
        [InlineData(17, 4, 8)]
        public static void TestFunc(double y, double x, double dx)
        {
            var dy = Calculus.DifferentiateAt(x, Func);
            Assert.Equal(y, dy.Magnitude);
            Assert.Equal(dx, dy.Derivatives[0]);
        }

        static Number Func(Number x) =>
            (x ^ 2) + 1;

        [Fact]
        public static void TestSample()
        {
            var y = Calculus.DifferentiateAt(2, 3, Sample);
            Assert.Equal(5.947664043757056, y.Magnitude);
            Assert.Equal(1.0013704652454263, y.Derivatives[1]);
        }

        static Number Sample(Number x1, Number x2) =>
            x1 * x2 - x2.SinDeg();
    }
}
