using System;
using System.Linq;
using System.Text;

namespace AutoDiffSharp
{
    /// <summary>
    /// A number type used for automatic differentiation.
    /// </summary>
    public readonly struct Number : IEquatable<Number>, IComparable<Number>
    {
        /// <summary>
        /// The computed magnitude.
        /// </summary>
        public readonly double Magnitude;

        /// <summary>
        /// The computed derivatives.
        /// </summary>
        public readonly Derivatives Derivatives;

        /// <summary>
        /// The magnitude.
        /// </summary>
        /// <param name="x"></param>
        public Number(double x) : this(x, 0)
        {
        }

        internal Number(double x, params double[] v) : this(x, new Derivatives(v))
        {
        }

        internal Number(double x, Derivatives diff)
        {
            this.Magnitude = x;
            this.Derivatives = diff;
        }

        /// <summary>
        /// Compute the sin in radians.
        /// </summary>
        /// <returns></returns>
        public Number Sin() =>
            new Number(Math.Sin(Magnitude), Derivatives * Math.Cos(Magnitude));

        /// <summary>
        /// Compute the sin in degrees.
        /// </summary>
        public Number SinDeg() =>
            new Number(Math.Sin(Magnitude * Math.PI / 180), Derivatives * Math.Cos(Magnitude * Math.PI / 180));

        /// <summary>
        /// Compute the cosine in radians.
        /// </summary>
        public Number Cos() =>
            new Number(Math.Cos(Magnitude), Derivatives * -Math.Sin(Magnitude));

        /// <summary>
        /// Compute the cosine in degrees.
        /// </summary>
        public Number CosDeg() =>
            new Number(Math.Cos(Magnitude * Math.PI / 180), Derivatives * -Math.Sin(Magnitude * Math.PI / 180));

        /// <summary>
        /// Compute the logarithm.
        /// </summary>
        public Number Log() =>
            new Number(Math.Log(Magnitude), Derivatives * (1 / Magnitude));

        /// <summary>
        /// Compute an exponentiation.
        /// </summary>
        /// <param name="k">The exponent.</param>
        public Number Pow(int k) =>
            new Number(Math.Pow(Magnitude, k), k * Math.Pow(Magnitude, k - 1) * Derivatives);

        /// <summary>
        /// Compute the absolute value.
        /// </summary>
        public Number Abs() =>
            new Number(Math.Abs(Magnitude), Derivatives * (Magnitude < 0 ? -1 : 1));

        /// <summary>
        /// Compute the exponential.
        /// </summary>
        public Number Exp() =>
            new Number(Math.Exp(Magnitude), Math.Exp(Magnitude) * Derivatives);

        /// <summary>
        /// <inheritdoc cref="IEquatable{T}.Equals(T)"/>
        /// </summary>
        public bool Equals(Number other) =>
            Magnitude == other.Magnitude && Derivatives.Equals(other.Derivatives);

        /// <summary>
        /// <inheritdoc cref="IComparable{T}.CompareTo(T)"/>
        /// </summary>
        public int CompareTo(Number other) =>
            Magnitude.CompareTo(other.Magnitude);

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override string ToString()
        {
            var buf = new StringBuilder().Append(Magnitude);
            for (int i = 0; i < Derivatives.Count; ++i)
                buf.Append(" + ").Append(Derivatives[i]).Append('ϵ').Append(i);
            return buf.ToString();
        }

        /// <summary>
        /// Negate the number.
        /// </summary>
        public static Number operator -(Number x) =>
            new Number(-x.Magnitude, -x.Derivatives);

        /// <summary>
        /// Add two numbers.
        /// </summary>
        public static Number operator +(Number lhs, Number rhs) =>
            new Number(lhs.Magnitude + rhs.Magnitude, lhs.Derivatives + rhs.Derivatives);

        /// <summary>
        /// Subtract two numbers.
        /// </summary>
        public static Number operator -(Number lhs, Number rhs) =>
            new Number(lhs.Magnitude - rhs.Magnitude, lhs.Derivatives - rhs.Derivatives);

        /// <summary>
        /// Multiply two numbers.
        /// </summary>
        public static Number operator *(Number lhs, Number rhs) =>
            new Number(lhs.Magnitude * rhs.Magnitude, lhs.Derivatives * rhs.Magnitude + rhs.Derivatives * lhs.Magnitude);

        /// <summary>
        /// Divide two numbers.
        /// </summary>
        public static Number operator /(Number lhs, Number rhs) =>
            new Number(lhs.Magnitude / rhs.Magnitude,
                      (lhs.Derivatives * rhs.Magnitude - lhs.Magnitude * rhs.Derivatives) / (rhs.Magnitude * rhs.Magnitude));

        /// <summary>
        /// Raise number to an exponent.
        /// </summary>
        public static Number operator ^(Number lhs, int rhs) =>
            lhs.Pow(rhs);

        /// <summary>
        /// Multiply two numbers.
        /// </summary>
        public static Number operator *(Number lhs, double rhs) =>
            new Number(lhs.Magnitude * rhs, lhs.Derivatives * rhs);

        /// <summary>
        /// Multiply two numbers.
        /// </summary>
        public static Number operator *(double lhs, Number rhs) =>
            rhs * lhs;

        /// <summary>
        /// Multiply two numbers.
        /// </summary>
        public static Number operator *(Number lhs, int rhs) =>
            new Number(lhs.Magnitude * rhs, lhs.Derivatives * rhs);

        /// <summary>
        /// Divide two numbers.
        /// </summary>
        public static Number operator /(Number lhs, double rhs) =>
            new Number(lhs.Magnitude / rhs, lhs.Derivatives / rhs);

        /// <summary>
        /// Divide two numbers.
        /// </summary>
        public static Number operator /(Number lhs, int rhs) =>
            new Number(lhs.Magnitude / rhs, lhs.Derivatives / rhs);

        /// <summary>
        /// Multiply two numbers.
        /// </summary>
        public static Number operator *(int lhs, Number rhs) =>
            rhs * lhs;

        /// <summary>
        /// Convert double to a Number.
        /// </summary>
        public static implicit operator Number(double x) =>
            new Number(x, 1);

        /// <summary>
        /// Convert int to a Number.
        /// </summary>
        public static implicit operator Number(int x) =>
            new Number(x, 1);
    }
}
