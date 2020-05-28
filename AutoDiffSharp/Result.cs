using System;
using System.Linq;
using System.Text;

namespace AutoDiffSharp
{
    /// <summary>
    /// A number type used for automatic differentiation.
    /// </summary>
    public readonly struct Result : IEquatable<Result>, IComparable<Result>
    {
        /// <summary>
        /// The computed magnitude.
        /// </summary>
        public readonly double Magnitude;

        /// <summary>
        /// The computed derivatives.
        /// </summary>
        readonly double[] derivatives;

        /// <summary>
        /// The magnitude.
        /// </summary>
        /// <param name="x"></param>
        public Result(double x) : this(x, 0)
        {
        }

        internal Result(double x, params double[] v)
        {
            this.Magnitude = x;
            this.derivatives = v;
        }

        /// <summary>
        /// The derivative at the given index.
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public double Derivative(int i) => derivatives[i];

        /// <summary>
        /// The number of derivatives.
        /// </summary>
        public int Count => derivatives.Length;

        /// <summary>
        /// <inheritdoc cref="IComparable{T}.CompareTo(T)"/>
        /// </summary>
        public int CompareTo(Result other) =>
            Magnitude.CompareTo(other.Magnitude);
        
        /// <summary>
        /// <inheritdoc cref="IEquatable{T}.Equals(T)"/>
        /// </summary>
        public bool Equals(Result other) =>
            Magnitude == other.Magnitude && derivatives.SequenceEqual(other.derivatives);

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override string ToString()
        {
            var buf = new StringBuilder().Append(Magnitude);
            for (int i = 0; i < derivatives.Length; ++i)
                buf.Append(" + ").Append(derivatives[i]).Append('ϵ').Append(i);
            return buf.ToString();
        }
    }
}
