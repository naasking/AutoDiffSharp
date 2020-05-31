using System;
using System.Diagnostics.SymbolStore;
using System.Linq;
using System.Text;

namespace AutoDiffSharp
{
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
        /// The derivatives continuation.
        /// </summary>
        internal readonly Action<double> Derivative;

        internal Codual(double x, Action<double> dx)
        {
            this.Magnitude = x;
            this.Derivative = dx;
        }

        /// <summary>
        /// Compute the sin in radians.
        /// </summary>
        /// <returns></returns>
        public Codual Sin()
        {
            var lhs = this;
            return new Codual(Math.Sin(Magnitude), dx => lhs.Derivative(dx * Math.Cos(lhs.Magnitude)));
        }

        /// <summary>
        /// Compute the sin in degrees.
        /// </summary>
        public Codual SinDeg()
        {
            var lhs = this;
            return new Codual(Math.Sin(Magnitude * Math.PI / 180), dx => lhs.Derivative(dx * Math.Cos(lhs.Magnitude * Math.PI / 180)));
        }

        /// <summary>
        /// Compute the cosine in radians.
        /// </summary>
        public Codual Cos()
        {
            var lhs = this;
            return new Codual(Math.Cos(Magnitude), dx => lhs.Derivative(dx * -Math.Sin(lhs.Magnitude)));
        }

        /// <summary>
        /// Compute the cosine in degrees.
        /// </summary>
        public Codual CosDeg()
        {
            var lhs = this;
            return new Codual(Math.Cos(Magnitude * Math.PI / 180), dx => lhs.Derivative(dx * -Math.Sin(lhs.Magnitude * Math.PI / 180)));
        }

        /// <summary>
        /// Compute the logarithm.
        /// </summary>
        public Codual Log()
        {
            var lhs = this;
            return new Codual(Math.Log(Magnitude), dx => lhs.Derivative(dx * (1 / lhs.Magnitude)));
        }

        /// <summary>
        /// Compute an exponentiation.
        /// </summary>
        /// <param name="k">The exponent.</param>
        public Codual Pow(int k)
        {
            var lhs = this;
            return new Codual(Math.Pow(Magnitude, k), dx => lhs.Derivative(k * Math.Pow(lhs.Magnitude, k - 1) * dx));
        }

        /// <summary>
        /// Compute the absolute value.
        /// </summary>
        public Codual Abs()
        {
            var lhs = this;
            return new Codual(Math.Abs(Magnitude), dx => lhs.Derivative(dx * (lhs.Magnitude < 0 ? -1 : 1)));
        }

        /// <summary>
        /// Compute the exponential.
        /// </summary>
        public Codual Exp()
        {
            var lhs = this;
            return new Codual(Math.Exp(Magnitude), dx => lhs.Derivative(Math.Exp(lhs.Magnitude) * dx));
        }

        /// <summary>
        /// <inheritdoc cref="IEquatable{T}.Equals(T)"/>
        /// </summary>
        public bool Equals(Codual other) =>
            Magnitude == other.Magnitude && Derivative.Equals(other.Derivative);

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
            new Codual(-x.Magnitude, dx => x.Derivative(-dx));

        /// <summary>
        /// Add two Coduals.
        /// </summary>
        public static Codual operator +(Codual lhs, Codual rhs) =>
            // if lhs == rhs, then propagate value only once to avoid exponential
            // blowup, ie. loop(N) { x = x + x } updates dx 2^N times in the naive case.
            new Codual(lhs.Magnitude + rhs.Magnitude,
                lhs.Derivative == rhs.Derivative
                ? new Action<double>(dx => lhs.Derivative(2 * dx))
                : dx =>
                {
                    lhs.Derivative(dx);
                    rhs.Derivative(dx);
                });

        /// <summary>
        /// Add two Coduals.
        /// </summary>
        public static Codual operator +(Codual lhs, double rhs) =>
            new Codual(lhs.Magnitude + rhs, dx => lhs.Derivative(dx));

        /// <summary>
        /// Add two Coduals.
        /// </summary>
        public static Codual operator +(double lhs, Codual rhs) =>
            rhs + lhs;

        /// <summary>
        /// Subtract two Coduals.
        /// </summary>
        public static Codual operator -(Codual lhs, Codual rhs) =>
            // if lhs == rhs, then contribution to derivative is zero
            new Codual(lhs.Magnitude - rhs.Magnitude, lhs.Derivative == rhs.Derivative
            ? new Action<double>(dx => { })
            : dx =>
            {
                lhs.Derivative(dx);
                rhs.Derivative(-dx);
            });

        /// <summary>
        /// Subtract two Coduals.
        /// </summary>
        public static Codual operator -(Codual lhs, double rhs) =>
            lhs + -rhs;

        /// <summary>
        /// Subtract two Coduals.
        /// </summary>
        public static Codual operator -(double lhs, Codual rhs) =>
            new Codual(lhs - rhs.Magnitude, dx => rhs.Derivative(-dx));

        /// <summary>
        /// Multiply two Coduals.
        /// </summary>
        public static Codual operator *(Codual lhs, Codual rhs) =>
            // if lhs == rhs, then propagate value only once to avoid exponential
            // blowup, ie. loop(N) { x = x * x } updates dx 2^N times in the naive case.
            new Codual(lhs.Magnitude * rhs.Magnitude, lhs.Derivative == rhs.Derivative
            ? new Action<double>(dx => lhs.Derivative(2 * dx * lhs.Magnitude))
            : dx =>
            {
                lhs.Derivative(dx * rhs.Magnitude);
                rhs.Derivative(dx * lhs.Magnitude);
            });

        /// <summary>
        /// Divide two Coduals.
        /// </summary>
        public static Codual operator /(Codual lhs, Codual rhs) =>
            // if lhs == rhs, then contribution to derivative is zero
            new Codual(lhs.Magnitude / rhs.Magnitude, lhs.Derivative == rhs.Derivative
            ? new Action<double>(dx => { })
            : dx =>
            {
                var d = rhs.Magnitude * rhs.Magnitude;
                lhs.Derivative(dx * rhs.Magnitude / d);
                rhs.Derivative(-dx * lhs.Magnitude / d);
            });

        /// <summary>
        /// Raise Reverse to an exponent.
        /// </summary>
        public static Codual operator ^(Codual lhs, int rhs) =>
            lhs.Pow(rhs);

        /// <summary>
        /// Multiply two Coduals.
        /// </summary>
        public static Codual operator *(Codual lhs, double rhs) =>
            new Codual(lhs.Magnitude * rhs, dx => lhs.Derivative(dx * rhs));

        /// <summary>
        /// Multiply two Coduals.
        /// </summary>
        public static Codual operator *(double lhs, Codual rhs) =>
            rhs * lhs;

        /// <summary>
        /// Multiply two Coduals.
        /// </summary>
        public static Codual operator *(Codual lhs, int rhs) =>
            new Codual(lhs.Magnitude * rhs, dx => lhs.Derivative(dx * rhs));

        /// <summary>
        /// Divide two Coduals.
        /// </summary>
        public static Codual operator /(Codual lhs, double rhs) =>
            new Codual(lhs.Magnitude / rhs, dx => lhs.Derivative(dx / rhs));

        /// <summary>
        /// Divide two Coduals.
        /// </summary>
        public static Codual operator /(Codual lhs, int rhs) =>
            new Codual(lhs.Magnitude / rhs, dx => lhs.Derivative(dx / rhs));

        /// <summary>
        /// Multiply two Coduals.
        /// </summary>
        public static Codual operator *(int lhs, Codual rhs) =>
            rhs * lhs;
    }
}
