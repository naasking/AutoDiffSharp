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
        public static Number DifferentiateAt(double x, Func<Number, Number> func) =>
            func(new Number(x, 1));

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static Number DifferentiateAt(double x0, double x1, Func<Number, Number, Number> func) =>
            func(new Number(x0, 1, 0), new Number(x1, 0, 1));

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="x2">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static Number DifferentiateAt(double x0, double x1, double x2, Func<Number, Number, Number, Number> func) =>
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
        public static Number DifferentiateAt(double x0, double x1, double x2, double x3, Func<Number, Number, Number, Number, Number> func) =>
            func(new Number(x0, 1, 0, 0, 0), new Number(x1, 0, 0, 1, 0), new Number(x2, 0, 0, 1, 0), new Number(x3, 0, 0, 0, 1));

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x">The function arguments.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>The function's value and its derivatives at the given point.</returns>
        public static Number DifferentiateAt(double[] x, Func<Number[], Number> func) =>
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
            x => DifferentiateAt(x, func);

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>A function that's the differentiation of <paramref name="func"/>.</returns>
        public static Func<double, double, Number> Differentiate(Func<Number, Number, Number> func) =>
            (x0, x1) => DifferentiateAt(x0, x1, func);

        /// <summary>
        /// Evaluate and a function at the given point and return its derivatives.
        /// </summary>
        /// <param name="x0">The function argument.</param>
        /// <param name="x1">The function argument.</param>
        /// <param name="x2">The function argument.</param>
        /// <param name="func">The function to evaluate.</param>
        /// <returns>A function that's the differentiation of <paramref name="func"/>.</returns>
        public static Func<double, double, double, Number> Differentiate(Func<Number, Number, Number, Number> func) =>
            (x0, x1, x2) => DifferentiateAt(x0, x1, x2, func);

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
            (x0, x1, x2, x3) => DifferentiateAt(x0, x1, x2, x3, func);
    }
}
