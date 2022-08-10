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
        struct Node
        {
            public Op Op;
            public int Id1;
            public double Dx1;
            public int Id2;
            public double Dx2;
            public override string ToString() => $"grad<{Op}>({Id1}, {Id2})";

            public void Gradient(double dx, double[] df)
            {
                switch (Op)
                {
                    case Op.Neg:
                        df[Id1] -= dx;
                        break;
                    case Op.Add:
                        df[Id1] += dx;
                        df[Id2] += dx;
                        break;
                    case Op.Sub:
                        df[Id1] += dx;
                        df[Id2] -= dx;
                        break;
                    case Op.Mul:
                        df[Id1] += dx * Dx2;
                        df[Id2] += dx * Dx1;
                        break;
                    case Op.Div:
                        df[Id1] += dx / Dx2;
                        df[Id2] -= Dx1 * dx / Dx2 / Dx2;
                        break;
                    case Op.Pow:
                        df[Id1] += dx * Id2 * Math.Pow(Dx1, Id2 - 1);
                        break;
                    case Op.Exp:
                        df[Id1] += dx * Math.Exp(Dx1);
                        break;
                    case Op.Abs:
                        df[Id1] += Math.Abs(dx);
                        break;
                    case Op.Log:
                        df[Id1] += dx / Dx1;
                        break;
                    case Op.Sin:
                        df[Id1] += dx * Math.Cos(Dx1);
                        break;
                    case Op.SinDeg:
                        df[Id1] += dx * Math.Cos(Dx1 * Math.PI / 180);
                        break;
                    case Op.Cos:
                        df[Id1] += dx * -Math.Sin(Dx1);
                        break;
                    case Op.CosDeg:
                        df[Id1] += dx * -Math.Sin(Dx1 * Math.PI / 180);
                        break;
                    default:
                        throw new InvalidOperationException("Unrecognized op: " + Op);
                }
            }
        }

        static double[] Propagate(int count, List<Node> q)
        {
            var dx = new double[q.Count];
            dx[dx.Length - 1] = 1;
            for (int i = dx.Length - 1; i >= count; --i)
                q[i].Gradient(dx[i], dx);
            return dx.Take(count).ToArray();
        }

        static (List<Node>, CreateNode) Create(int i)
        {
            var q = new List<Node>();
            while (i-- > 0)
                q.Add(new Node());
            int df(int id1, double dx1, int id2, double dx2, Op op)
            {
                q.Add(new Node { Id1 = id1, Dx1 = dx1, Id2 = id2, Dx2 = dx2, Op = op });
                return q.Count - 1;
            }
            return (q, df);
        }

        /// <summary>
        /// Compute the derivative for the given value.
        /// </summary>
        /// <param name="x">The value at which to compute the derivative.</param>
        /// <param name="f">The function to differentiate.</param>
        /// <returns>The value and derivative at the given point.</returns>
        public static Result DerivativeAt(double x, Func<Codual, Codual> f)
        {
            var (q, df) = Create(1);
            var y = f(new Codual(x, df, 0));
            return new Result(y.Magnitude, Propagate(1, q));
        }

        /// <summary>
        /// Compute the derivative for the given values.
        /// </summary>
        /// <param name="x0">The value at which to compute the derivative.</param>
        /// <param name="x1">The value at which to compute the derivative.</param>
        /// <param name="f">The function to differentiate.</param>
        /// <returns>The value and derivative at the given point.</returns>

        public static Result DerivativeAt(double x0, double x1, Func<Codual, Codual, Codual> f)
        {
            var (q, df) = Create(2);
            var y = f(new Codual(x0, df, 0),
                      new Codual(x1, df, 1));
            return new Result(y.Magnitude, Propagate(2, q));
        }

        /// <summary>
        /// Compute the derivative for the given values.
        /// </summary>
        /// <param name="x0">The value at which to compute the derivative.</param>
        /// <param name="x1">The value at which to compute the derivative.</param>
        /// <param name="x2">The value at which to compute the derivative.</param>
        /// <param name="f">The function to differentiate.</param>
        /// <returns>The value and derivative at the given point.</returns>
        public static Result DerivativeAt(double x0, double x1, double x2, Func<Codual, Codual, Codual, Codual> f)
        {
            var (q, df) = Create(3);
            var y = f(new Codual(x0, df, 0),
                      new Codual(x1, df, 1),
                      new Codual(x2, df, 2));
            return new Result(y.Magnitude, Propagate(3, q));
        }

        /// <summary>
        /// Compute the derivative for the given values.
        /// </summary>
        /// <param name="x">The values at which to compute the derivative.</param>
        /// <param name="f">The function to differentiate.</param>
        /// <returns>The value and derivative at the given point.</returns>
        public static Result DerivativeAt(double[] x, Func<Codual[], Codual> f)
        {
            var (q, df) = Create(x.Length);
            var args = x.Select((z, i) => new Codual(z, df, i)).ToArray();
            var y = f(args);
            return new Result(y.Magnitude, Propagate(x.Length, q));
        }

        /// <summary>
        /// Differentiate a function.
        /// </summary>
        /// <param name="f">The function to differentiate.</param>
        /// <returns>The differentiation of the given function.</returns>
        public static Func<double, Result> Differentiate(this Func<Codual, Codual> f) =>
            x => DerivativeAt(x, f);

        /// <summary>
        /// Differentiate a function.
        /// </summary>
        /// <param name="f">The function to differentiate.</param>
        /// <returns>The differentiation of the given function.</returns>
        public static Func<double, double, Result> Differentiate(this Func<Codual, Codual, Codual> f) =>
            (x0, x1) => DerivativeAt(x0, x1, f);

        /// <summary>
        /// Differentiate a function.
        /// </summary>
        /// <param name="f">The function to differentiate.</param>
        /// <returns>The differentiation of the given function.</returns>
        public static Func<double, double, double, Result> Differentiate(Func<Codual, Codual, Codual, Codual> f) =>
            (x0, x1, x2) => DerivativeAt(x0, x1, x2, f);

        /// <summary>
        /// Differentiate a function.
        /// </summary>
        /// <param name="f">The function to differentiate.</param>
        /// <returns>The differentiation of the given function.</returns>
        public static Func<double[], Result> Differentiate(this Func<Codual[], Codual> f) =>
            x => DerivativeAt(x, f);

        #endregion
    }
}
