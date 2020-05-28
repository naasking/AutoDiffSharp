using System;
using System.Linq;
using System.Text;

namespace AutoDiffSharp
{
    /// <summary>
    /// A numeric type for forward-mode automatic differentiation.
    /// </summary>
    public readonly struct Dual : IEquatable<Dual>, IComparable<Dual>
    {
        /// <summary>
        /// The computed magnitude.
        /// </summary>
        public readonly double Magnitude;

        /// <summary>
        /// The computed derivative.
        /// </summary>
        public readonly double Derivative;

        /// <summary>
        /// The magnitude.
        /// </summary>
        /// <param name="x"></param>
        public Dual(double x) : this(x, 0)
        {
        }

        internal Dual(double x, double dx)
        {
            this.Magnitude = x;
            this.Derivative = dx;
        }

        /// <summary>
        /// Compute the sin in radians.
        /// </summary>
        /// <returns></returns>
        public Dual Sin() =>
            new Dual(Math.Sin(Magnitude), Derivative * Math.Cos(Magnitude));

        /// <summary>
        /// Compute the sin in degrees.
        /// </summary>
        public Dual SinDeg() =>
            new Dual(Math.Sin(Magnitude * Math.PI / 180), Derivative * Math.Cos(Magnitude * Math.PI / 180));

        /// <summary>
        /// Compute the cosine in radians.
        /// </summary>
        public Dual Cos() =>
            new Dual(Math.Cos(Magnitude), Derivative * -Math.Sin(Magnitude));

        /// <summary>
        /// Compute the cosine in degrees.
        /// </summary>
        public Dual CosDeg() =>
            new Dual(Math.Cos(Magnitude * Math.PI / 180), Derivative * -Math.Sin(Magnitude * Math.PI / 180));

        /// <summary>
        /// Compute the logarithm.
        /// </summary>
        public Dual Log() =>
            new Dual(Math.Log(Magnitude), Derivative * (1 / Magnitude));

        /// <summary>
        /// Compute an exponentiation.
        /// </summary>
        /// <param name="k">The exponent.</param>
        public Dual Pow(int k) =>
            new Dual(Math.Pow(Magnitude, k), k * Math.Pow(Magnitude, k - 1) * Derivative);

        /// <summary>
        /// Compute the absolute value.
        /// </summary>
        public Dual Abs() =>
            new Dual(Math.Abs(Magnitude), Derivative * (Magnitude < 0 ? -1 : 1));

        /// <summary>
        /// Compute the exponential.
        /// </summary>
        public Dual Exp() =>
            new Dual(Math.Exp(Magnitude), Math.Exp(Magnitude) * Derivative);

        /// <summary>
        /// <inheritdoc cref="IEquatable{T}.Equals(T)"/>
        /// </summary>
        public bool Equals(Dual other) =>
            Magnitude == other.Magnitude && Derivative.Equals(other.Derivative);

        /// <summary>
        /// <inheritdoc cref="IComparable{T}.CompareTo(T)"/>
        /// </summary>
        public int CompareTo(Dual other) =>
            Magnitude.CompareTo(other.Magnitude);

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override string ToString() => $"{Magnitude} + {Derivative}ϵ";

        /// <summary>
        /// Negate the Dual.
        /// </summary>
        public static Dual operator -(Dual x) =>
            new Dual(-x.Magnitude, -x.Derivative);

        /// <summary>
        /// Add two Duals.
        /// </summary>
        public static Dual operator +(Dual lhs, Dual rhs) =>
            new Dual(lhs.Magnitude + rhs.Magnitude, lhs.Derivative + rhs.Derivative);

        /// <summary>
        /// Add two Duals.
        /// </summary>
        public static Dual operator +(Dual lhs, double rhs) =>
            new Dual(lhs.Magnitude + rhs, lhs.Derivative);

        /// <summary>
        /// Add two Duals.
        /// </summary>
        public static Dual operator +(double lhs, Dual rhs) =>
            rhs + lhs;

        /// <summary>
        /// Subtract two Duals.
        /// </summary>
        public static Dual operator -(Dual lhs, Dual rhs) =>
            new Dual(lhs.Magnitude - rhs.Magnitude, lhs.Derivative - rhs.Derivative);

        /// <summary>
        /// Subtract two Duals.
        /// </summary>
        public static Dual operator -(Dual lhs, double rhs) =>
            lhs + -rhs;

        /// <summary>
        /// Subtract two Duals.
        /// </summary>
        public static Dual operator -(double lhs, Dual rhs) =>
            new Dual(lhs - rhs.Magnitude, -rhs.Derivative);

        /// <summary>
        /// Multiply two Duals.
        /// </summary>
        public static Dual operator *(Dual lhs, Dual rhs) =>
            new Dual(lhs.Magnitude * rhs.Magnitude, lhs.Derivative * rhs.Magnitude + rhs.Derivative * lhs.Magnitude);

        /// <summary>
        /// Divide two Duals.
        /// </summary>
        public static Dual operator /(Dual lhs, Dual rhs) =>
            new Dual(lhs.Magnitude / rhs.Magnitude,
                      (lhs.Derivative * rhs.Magnitude - lhs.Magnitude * rhs.Derivative) / (rhs.Magnitude * rhs.Magnitude));

        /// <summary>
        /// Raise Dual to an exponent.
        /// </summary>
        public static Dual operator ^(Dual lhs, int rhs) =>
            lhs.Pow(rhs);

        /// <summary>
        /// Multiply two Duals.
        /// </summary>
        public static Dual operator *(Dual lhs, double rhs) =>
            new Dual(lhs.Magnitude * rhs, lhs.Derivative * rhs);

        /// <summary>
        /// Multiply two Duals.
        /// </summary>
        public static Dual operator *(double lhs, Dual rhs) =>
            rhs * lhs;

        /// <summary>
        /// Multiply two Duals.
        /// </summary>
        public static Dual operator *(Dual lhs, int rhs) =>
            new Dual(lhs.Magnitude * rhs, lhs.Derivative * rhs);

        /// <summary>
        /// Divide two Duals.
        /// </summary>
        public static Dual operator /(Dual lhs, double rhs) =>
            new Dual(lhs.Magnitude / rhs, lhs.Derivative / rhs);

        /// <summary>
        /// Divide two Duals.
        /// </summary>
        public static Dual operator /(Dual lhs, int rhs) =>
            new Dual(lhs.Magnitude / rhs, lhs.Derivative / rhs);

        /// <summary>
        /// Multiply two Duals.
        /// </summary>
        public static Dual operator *(int lhs, Dual rhs) =>
            rhs * lhs;
    }
}
