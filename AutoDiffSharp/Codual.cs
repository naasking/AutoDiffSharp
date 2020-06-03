using System;
using System.Collections.Generic;
using System.Diagnostics.SymbolStore;
using System.Linq;
using System.Text;

namespace AutoDiffSharp
{
    internal delegate int CreateNode(int id1, int id2, Func<double, (double, double)> k);

    /// <summary>
    /// The numeric type for reverse-mode automatic differentiation.
    /// </summary>
    public readonly struct Codual : IEquatable<Codual>, IComparable<Codual>
    {
        /// <summary>
        /// The computed magnitude.
        /// </summary>
        public readonly double Magnitude;

        /// <summary>
        /// The unique identifier for this node.
        /// </summary>
        internal readonly int Id;

        /// <summary>
        /// The 
        /// </summary>
        internal readonly CreateNode CreateNode;

        internal const int IGNORE = 0;
        const double NONE = 0;

        internal Codual(double x, CreateNode dx, int id)
        {
            this.Magnitude = x;
            this.Id = id;
            this.CreateNode = dx;
        }

        /// <summary>
        /// Compute the sin in radians.
        /// </summary>
        /// <returns></returns>
        public Codual Sin()
        {
            var lhs = this;
            return new Codual(Math.Sin(Magnitude), CreateNode,
                              CreateNode(Id, IGNORE, dx => (dx * Math.Cos(lhs.Magnitude), NONE)));
        }

        /// <summary>
        /// Compute the sin in degrees.
        /// </summary>
        public Codual SinDeg()
        {
            var lhs = this;
            return new Codual(Math.Sin(Magnitude * Math.PI / 180), CreateNode,
                              CreateNode(Id, IGNORE, dx => (dx * Math.Cos(lhs.Magnitude * Math.PI / 180), NONE)));
        }

        /// <summary>
        /// Compute the cosine in radians.
        /// </summary>
        public Codual Cos()
        {
            var lhs = this;
            return new Codual(Math.Cos(Magnitude), CreateNode,
                              CreateNode(Id, IGNORE, dx => (dx * -Math.Sin(lhs.Magnitude), NONE)));
        }

        /// <summary>
        /// Compute the cosine in degrees.
        /// </summary>
        public Codual CosDeg()
        {
            var lhs = this;
            return new Codual(Math.Cos(Magnitude * Math.PI / 180), CreateNode,
                              CreateNode(Id, IGNORE, dx => (dx * -Math.Sin(lhs.Magnitude * Math.PI / 180), NONE)));
        }

        /// <summary>
        /// Compute the logarithm.
        /// </summary>
        public Codual Log()
        {
            var lhs = this;
            return new Codual(Math.Log(Magnitude), CreateNode,
                              CreateNode(Id, IGNORE, dx => (dx * -Math.Sin(lhs.Magnitude * Math.PI / 180), NONE)));
        }

        /// <summary>
        /// Compute an exponentiation.
        /// </summary>
        /// <param name="k">The exponent.</param>
        public Codual Pow(int k)
        {
            var lhs = this;
            return new Codual(Math.Pow(Magnitude, k), CreateNode,
                              CreateNode(Id, IGNORE, dx => (dx * k * Math.Pow(lhs.Magnitude, k - 1), NONE)));
        }

        /// <summary>
        /// Compute the absolute value.
        /// </summary>
        public Codual Abs()
        {
            var lhs = this;
            return new Codual(Math.Abs(Magnitude), CreateNode,
                              CreateNode(Id, IGNORE, dx => (Math.Abs(dx), NONE)));
        }

        /// <summary>
        /// Compute the exponential.
        /// </summary>
        public Codual Exp()
        {
            var lhs = this;
            return new Codual(Math.Exp(Magnitude), CreateNode,
                              CreateNode(Id, IGNORE, dx => (dx * Math.Exp(lhs.Magnitude), NONE)));
        }

        /// <summary>
        /// <inheritdoc cref="IEquatable{T}.Equals(T)"/>
        /// </summary>
        public bool Equals(Codual other) =>
            Id == other.Id;

        /// <summary>
        /// <inheritdoc cref="IComparable{T}.CompareTo(T)"/>
        /// </summary>
        public int CompareTo(Codual other) =>
            Magnitude.CompareTo(other.Magnitude);

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override string ToString() => $"{Magnitude} + Xϵ";

        /// <summary>
        /// Negate the Reverse.
        /// </summary>
        public static Codual operator -(Codual x) =>
            new Codual(-x.Magnitude, x.CreateNode, x.CreateNode(x.Id, IGNORE, dx => (-dx, NONE)));

        /// <summary>
        /// Add two Coduals.
        /// </summary>
        public static Codual operator +(Codual lhs, Codual rhs) =>
            new Codual(lhs.Magnitude + rhs.Magnitude, lhs.CreateNode,
                       lhs.CreateNode(lhs.Id, rhs.Id, dx => (dx, dx)));

        /// <summary>
        /// Add two Coduals.
        /// </summary>
        public static Codual operator +(Codual lhs, double rhs) =>
            new Codual(lhs.Magnitude + rhs, lhs.CreateNode,
                       lhs.CreateNode(lhs.Id, IGNORE, dx => (dx, NONE)));

        /// <summary>
        /// Subtract two Coduals.
        /// </summary>
        public static Codual operator -(Codual lhs, Codual rhs) =>
            lhs + -rhs;

        /// <summary>
        /// Subtract two Coduals.
        /// </summary>
        public static Codual operator -(Codual lhs, double rhs) =>
            lhs + -rhs;

        ///// <summary>
        ///// Subtract two Coduals.
        ///// </summary>
        //public static Codual operator -(double lhs, Codual rhs) =>
        //    new Codual(lhs - rhs.Magnitude, dx => rhs.Derivative(-dx));

        /// <summary>
        /// Multiply two Coduals.
        /// </summary>
        public static Codual operator *(Codual lhs, Codual rhs) =>
            new Codual(lhs.Magnitude* rhs.Magnitude, lhs.CreateNode,
                       lhs.CreateNode(lhs.Id, rhs.Id, dx => (dx * rhs.Magnitude, dx * lhs.Magnitude)));

        /// <summary>
        /// Divide two Coduals.
        /// </summary>
        public static Codual operator /(Codual lhs, Codual rhs) =>
            new Codual(lhs.Magnitude / rhs.Magnitude, lhs.CreateNode,
                       lhs.CreateNode(lhs.Id, rhs.Id, dx => (dx / rhs.Magnitude, -dx / lhs.Magnitude)));

        /// <summary>
        /// Raise Codual to an exponent.
        /// </summary>
        public static Codual operator ^(Codual lhs, int rhs) =>
            lhs.Pow(rhs);

        /// <summary>
        /// Multiply two Coduals.
        /// </summary>
        public static Codual operator *(Codual lhs, double rhs) =>
            new Codual(lhs.Magnitude * rhs, lhs.CreateNode,
                       lhs.CreateNode(lhs.Id, IGNORE, dx => (dx * rhs, NONE)));

        /// <summary>
        /// Multiply two Coduals.
        /// </summary>
        public static Codual operator *(Codual lhs, int rhs) =>
            new Codual(lhs.Magnitude * rhs, lhs.CreateNode,
                       lhs.CreateNode(lhs.Id, IGNORE, dx => (dx * rhs, NONE)));

        /// <summary>
        /// Divide two Coduals.
        /// </summary>
        public static Codual operator /(Codual lhs, double rhs) =>
            new Codual(lhs.Magnitude / rhs, lhs.CreateNode,
                       lhs.CreateNode(lhs.Id, IGNORE, dx => (dx / rhs, NONE)));

        /// <summary>
        /// Divide two Coduals.
        /// </summary>
        public static Codual operator /(Codual lhs, int rhs) =>
            new Codual(lhs.Magnitude / rhs, lhs.CreateNode,
                       lhs.CreateNode(lhs.Id, IGNORE, dx => (dx / rhs, NONE)));
    }
}
