using System;
using System.Collections.Generic;
using System.Diagnostics.SymbolStore;
using System.Linq;
using System.Text;

namespace AutoDiffSharp
{
    /// <summary>
    /// Record a node in the trace.
    /// </summary>
    /// <returns>The index of the new node.</returns>
    internal delegate int CreateNode(int id1, double dx1, int id2, double dx2, Op op);

    /// <summary>
    /// Math operator.
    /// </summary>
    internal enum Op
    {
        Var = 0,
        Neg,
        Add,
        Sub,
        Mul,
        Div,
        Pow,
        Exp,
        Log,
        Abs,
        Sin,
        SinDeg,
        Cos,
        CosDeg,
    }

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
        /// The node constructor.
        /// </summary>
        internal readonly CreateNode CreateNode;

        const int IGNORE = 0;
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
        public Codual Sin() =>
            new Codual(Math.Sin(Magnitude), CreateNode,
                       CreateNode(Id, Magnitude, IGNORE, NONE, Op.Sin));

        /// <summary>
        /// Compute the sin in degrees.
        /// </summary>
        public Codual SinDeg() =>
            new Codual(Math.Sin(Magnitude * Math.PI / 180), CreateNode,
                       CreateNode(Id, Magnitude, IGNORE, NONE, Op.SinDeg));

        /// <summary>
        /// Compute the cosine in radians.
        /// </summary>
        public Codual Cos() =>
            new Codual(Math.Cos(Magnitude), CreateNode,
                       CreateNode(Id, Magnitude, IGNORE, NONE, Op.Cos));

        /// <summary>
        /// Compute the cosine in degrees.
        /// </summary>
        public Codual CosDeg() =>
            new Codual(Math.Cos(Magnitude * Math.PI / 180), CreateNode,
                       CreateNode(Id, Magnitude, IGNORE, NONE, Op.CosDeg));

        /// <summary>
        /// Compute the logarithm.
        /// </summary>
        public Codual Log() =>
            new Codual(Math.Log(Magnitude), CreateNode,
                       CreateNode(Id, Magnitude, IGNORE, NONE, Op.Log));

        /// <summary>
        /// Compute an exponentiation.
        /// </summary>
        /// <param name="k">The exponent.</param>
        public Codual Pow(int k) =>
            new Codual(Math.Pow(Magnitude, k), CreateNode,
                       CreateNode(Id, Magnitude, k, NONE, Op.Pow));

        /// <summary>
        /// Compute the absolute value.
        /// </summary>
        public Codual Abs() =>
            new Codual(Math.Abs(Magnitude), CreateNode,
                       CreateNode(Id, NONE, IGNORE, NONE, Op.Abs));

        /// <summary>
        /// Compute the exponential.
        /// </summary>
        public Codual Exp() =>
            new Codual(Math.Exp(Magnitude), CreateNode,
                       CreateNode(Id, Magnitude, IGNORE, NONE, Op.Exp));

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
            new Codual(-x.Magnitude, x.CreateNode, x.CreateNode(x.Id, NONE, IGNORE, NONE, Op.Neg));

        /// <summary>
        /// Add two Coduals.
        /// </summary>
        public static Codual operator +(Codual lhs, Codual rhs) =>
            new Codual(lhs.Magnitude + rhs.Magnitude, lhs.CreateNode,
                       lhs.CreateNode(lhs.Id, NONE, rhs.Id, NONE, Op.Add));

        /// <summary>
        /// Add two Coduals.
        /// </summary>
        public static Codual operator +(Codual lhs, double rhs) =>
            new Codual(lhs.Magnitude + rhs, lhs.CreateNode,
                       lhs.CreateNode(lhs.Id, NONE, IGNORE, NONE, Op.Add));

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
                       lhs.CreateNode(lhs.Id, lhs.Magnitude, rhs.Id, rhs.Magnitude, Op.Mul));

        /// <summary>
        /// Divide two Coduals.
        /// </summary>
        public static Codual operator /(Codual lhs, Codual rhs) =>
            new Codual(lhs.Magnitude / rhs.Magnitude, lhs.CreateNode,
                       lhs.CreateNode(lhs.Id, lhs.Magnitude, rhs.Id, rhs.Magnitude, Op.Div));

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
                       lhs.CreateNode(lhs.Id, NONE, IGNORE, rhs, Op.Mul));

        /// <summary>
        /// Multiply two Coduals.
        /// </summary>
        public static Codual operator *(Codual lhs, int rhs) =>
            new Codual(lhs.Magnitude * rhs, lhs.CreateNode,
                       lhs.CreateNode(lhs.Id, NONE, IGNORE, rhs, Op.Mul));

        /// <summary>
        /// Divide two Coduals.
        /// </summary>
        public static Codual operator /(Codual lhs, double rhs) =>
            new Codual(lhs.Magnitude / rhs, lhs.CreateNode,
                       lhs.CreateNode(lhs.Id, NONE, IGNORE, rhs, Op.Div));

        /// <summary>
        /// Divide two Coduals.
        /// </summary>
        public static Codual operator /(Codual lhs, int rhs) =>
            new Codual(lhs.Magnitude / rhs, lhs.CreateNode,
                       lhs.CreateNode(lhs.Id, NONE, IGNORE, rhs, Op.Div));
    }
}
