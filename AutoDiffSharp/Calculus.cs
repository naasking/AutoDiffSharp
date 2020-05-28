using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;

namespace AutoDiffSharp
{
    /// <summary>
    /// Implementations of calculus.
    /// </summary>
    public static class Calculus
    {
        #region Number vectors
        /// <summary>
        /// Evaluate and a function at the given point without differentiating.
        /// </summary>
        /// <param name="x">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static double Apply(double x, Func<Number, Number> func) =>
            func(new Number(x, 0)).Magnitude;

        /// <summary>
        /// Evaluate and a function at the given point without differentiating.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static double Apply(double x0, double x1, Func<Number, Number, Number> func) =>
            func(new Number(x0, 0), new Number(x1, 0)).Magnitude;

        /// <summary>
        /// Evaluate and a function at the given point without differentiating.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="x2">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static double Apply(double x0, double x1, double x2, Func<Number, Number, Number, Number> func) =>
            func(new Number(x0, 0), new Number(x1, 0), new Number(x2, 0)).Magnitude;

        /// <summary>
        /// Evaluate and a function at the given point without differentiating.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="x2">The function argument.</param>
        /// <param name="x3">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static double Apply(double x0, double x1, double x2, double x3, Func<Number, Number, Number, Number, Number> func) =>
            func(new Number(x0, 0), new Number(x1, 0), new Number(x2, 0), new Number(x3, 0)).Magnitude;

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static Number DerivativeAt(double x, Func<Number, Number> func) =>
            func(new Number(x, 1));

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static Number DerivativeAt(double x0, double x1, Func<Number, Number, Number> func) =>
            func(new Number(x0, 1, 0), new Number(x1, 0, 1));

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="x2">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static Number DerivativeAt(double x0, double x1, double x2, Func<Number, Number, Number, Number> func) =>
            func(new Number(x0, 1, 0, 0), new Number(x1, 0, 0, 1), new Number(x2, 0, 0, 1));

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="x2">The function argument.</param>
        /// <param name="x3">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static Number DerivativeAt(double x0, double x1, double x2, double x3, Func<Number, Number, Number, Number, Number> func) =>
            func(new Number(x0, 1, 0, 0, 0), new Number(x1, 0, 0, 1, 0), new Number(x2, 0, 0, 1, 0), new Number(x3, 0, 0, 0, 1));

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x">The function arguments.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static Number DerivativeAt(double[] x, Func<Number[], Number> func) =>
            func(x.Select((y, i) =>
            {
                var v = new double[x.Length];
                v[i] = 1;
                return new Number(y, new Derivatives(v));
            }).ToArray());


        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>A function that's the differentiation of <paramref name="func"/>.</returns>
        public static Func<double, Number> Differentiate(Func<Number, Number> func) =>
            x => DerivativeAt(x, func);

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>A function that's the differentiation of <paramref name="func"/>.</returns>
        public static Func<double, double, Number> Differentiate(Func<Number, Number, Number> func) =>
            (x0, x1) => DerivativeAt(x0, x1, func);

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="x2">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>A function that's the differentiation of <paramref name="func"/>.</returns>
        public static Func<double, double, double, Number> Differentiate(Func<Number, Number, Number, Number> func) =>
            (x0, x1, x2) => DerivativeAt(x0, x1, x2, func);

        /// <summary>
        /// Differentiate a function.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="x2">The function argument.</param>
        /// <param name="x3">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>A function that's the differentiation of <paramref name="func"/>.</returns>
        public static Func<double, double, double, double, Number> Differentiate(Func<Number, Number, Number, Number, Number> func) =>
            (x0, x1, x2, x3) => DerivativeAt(x0, x1, x2, x3, func);

        #endregion

        #region Dual numbers
        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static Dual DerivativeAt(double x, Func<Dual, Dual> func) =>
            func(new Dual(x, 1));

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static Dual DerivativeX0At(double x0, double x1, Func<Dual, Dual, Dual> func) =>
            func(new Dual(x0, 1), new Dual(x1, 0));

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static Dual DerivativeX1At(double x0, double x1, Func<Dual, Dual, Dual> func) =>
            func(new Dual(x0, 0), new Dual(x1, 1));

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="x2">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static Dual DerivativeX0At(double x0, double x1, double x2, Func<Dual, Dual, Dual, Dual> func) =>
            func(new Dual(x0, 1), new Dual(x1, 0), new Dual(x2, 0));

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="x2">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static Dual DerivativeX1At(double x0, double x1, double x2, Func<Dual, Dual, Dual, Dual> func) =>
            func(new Dual(x0, 0), new Dual(x1, 1), new Dual(x2, 0));

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="x2">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static Dual DerivativeX2At(double x0, double x1, double x2, Func<Dual, Dual, Dual, Dual> func) =>
            func(new Dual(x0, 0), new Dual(x1, 0), new Dual(x2, 1));
        #endregion

        #region CoDual Numbers
        /// <summary>
        /// Compute the derivative for the given value.
        /// </summary>
        /// <param name="x">The value at which to compute the derivative.</param>
        /// <param name="f">The function to differentiate.</param>
        /// <returns>The value and derivative at the given point.</returns>
        public static Dual DerivativeAt(double x, Func<Codual, Codual> f)
        {
            var dx = 0.0;
            var y = f(new Codual(x, dy => dx += dy));
            y.Derivative(1);
            return new Dual(y.Magnitude, dx);
        }
        /// <summary>
        /// Compute the derivative for the given values.
        /// </summary>
        /// <param name="x0">The value at which to compute the derivative.</param>
        /// <param name="x1">The value at which to compute the derivative.</param>
        /// <param name="f">The function to differentiate.</param>
        /// <returns>The value and derivative at the given point.</returns>

        public static Number DerivativeAt(double x0, double x1, Func<Codual, Codual, Codual> f)
        {
            double dx0 = 0, dx1 = 0;
            var y = f(new Codual(x0, dy => dx0 += dy),
                      new Codual(x1, dy => dx1 += dy));
            y.Derivative(1);
            return new Number(y.Magnitude, dx0, dx1);
        }

        /// <summary>
        /// Compute the derivative for the given values.
        /// </summary>
        /// <param name="x0">The value at which to compute the derivative.</param>
        /// <param name="x1">The value at which to compute the derivative.</param>
        /// <param name="x2">The value at which to compute the derivative.</param>
        /// <param name="f">The function to differentiate.</param>
        /// <returns>The value and derivative at the given point.</returns>
        public static Number DerivativeAt(double x0, double x1, double x2, Func<Codual, Codual, Codual, Codual> f)
        {
            double dx0 = 0, dx1 = 0, dx2 = 0;
            var y = f(new Codual(x0, dy => dx0 += dy),
                      new Codual(x1, dy => dx1 += dy),
                      new Codual(x2, dy => dx2 += dy));
            y.Derivative(1);
            return new Number(y.Magnitude, dx0, dx1, dx2);
        }

        /// <summary>
        /// Compute the derivative for the given values.
        /// </summary>
        /// <param name="x">The values at which to compute the derivative.</param>
        /// <param name="f">The function to differentiate.</param>
        /// <returns>The value and derivative at the given point.</returns>
        public static Number DerivativeAt(double[] x, Func<Codual[], Codual> f)
        {
            var dx = new double[x.Length];
            var args = x.Select((z, i) => new Codual(z, dz => dx[i] += dz)).ToArray();
            var y = f(args);
            y.Derivative(1);
            return new Number(y.Magnitude, dx);
        }

        /// <summary>
        /// Differentiate a function.
        /// </summary>
        /// <param name="f">The function to differentiate.</param>
        /// <returns>The differentiation of the given function.</returns>
        public static Func<double, Dual> Differentiate(this Func<Codual, Codual> f) =>
            x => DerivativeAt(x, f);

        /// <summary>
        /// Differentiate a function.
        /// </summary>
        /// <param name="f">The function to differentiate.</param>
        /// <returns>The differentiation of the given function.</returns>
        public static Func<double, double, Number> Differentiate(this Func<Codual, Codual, Codual> f) =>
            (x0, x1) => DerivativeAt(x0, x1, f);

        /// <summary>
        /// Differentiate a function.
        /// </summary>
        /// <param name="f">The function to differentiate.</param>
        /// <returns>The differentiation of the given function.</returns>
        public static Func<double, double, double, Number> Differentiate(Func<Codual, Codual, Codual, Codual> f) =>
            (x0, x1, x2) => DerivativeAt(x0, x1, x2, f);

        /// <summary>
        /// Differentiate a function.
        /// </summary>
        /// <param name="f">The function to differentiate.</param>
        /// <returns>The differentiation of the given function.</returns>
        public static Func<double[], Number> Differentiate(this Func<Codual[], Codual> f) =>
            x => DerivativeAt(x, f);

        #endregion
    }
}
