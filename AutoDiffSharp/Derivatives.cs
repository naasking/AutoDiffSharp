using System;
using System.Linq;
using System.Collections.Generic;

namespace AutoDiffSharp
{
    /// <summary>
    /// Encapsulates computed derivatives.
    /// </summary>
    public readonly struct Derivatives : IEquatable<Derivatives>
    {
        readonly double[] vector;

        internal Derivatives(double[] vector) =>
            this.vector = vector;

        /// <summary>
        /// The number of derivatives.
        /// </summary>
        public int Count => vector?.Length ?? 0;

        /// <summary>
        /// Access a specific derivative.
        /// </summary>
        /// <param name="argumentIndex">The index of the argument to the function.</param>
        /// <returns>The derivative of the given argument.</returns>
        public double this[int argumentIndex] => vector[argumentIndex];

        /// <summary>
        /// <inheritdoc cref="IEquatable{T}.Equals(T)"/>
        /// </summary>
        public bool Equals(Derivatives other) =>
            true == vector?.SequenceEqual(other.vector);

        /// <summary>
        /// Subtract derivatives.
        /// </summary>
        public static Derivatives operator -(Derivatives lhs)
        {
            var v = new double[lhs.Count];
            for (int i = 0; i < lhs.vector.Length; ++i)
                v[i] = -lhs.vector[i];
            return new Derivatives(v);
        }

        /// <summary>
        /// Add derivatives.
        /// </summary>
        public static Derivatives operator +(Derivatives lhs, Derivatives rhs)
        {
            var v = new double[lhs.Count];
            for (int i = 0; i < lhs.vector.Length; ++i)
                v[i] = lhs.vector[i] + rhs.vector[i];
            return new Derivatives(v);
        }

        /// <summary>
        /// Subtract derivatives.
        /// </summary>
        public static Derivatives operator -(Derivatives lhs, Derivatives rhs)
        {
            var v = new double[lhs.Count];
            for (int i = 0; i < lhs.vector.Length; ++i)
                v[i] = lhs.vector[i] - rhs.vector[i];
            return new Derivatives(v);
        }

        /// <summary>
        /// Multiply derivatives.
        /// </summary>
        public static Derivatives operator *(Derivatives lhs, Derivatives rhs)
        {
            var v = new double[lhs.Count];
            for (int i = 0; i < lhs.vector.Length; ++i)
                v[i] = lhs.vector[i] * rhs.vector[i];
            return new Derivatives(v);
        }

        /// <summary>
        /// Add a number to derivatives.
        /// </summary>
        public static Derivatives operator +(Derivatives lhs, double rhs)
        {
            var v = new double[lhs.Count];
            for (int i = 0; i < lhs.vector.Length; ++i)
                v[i] = lhs.vector[i] + rhs;
            return new Derivatives(v);
        }

        /// <summary>
        /// Subtracta number from derivatives.
        /// </summary>
        public static Derivatives operator -(Derivatives lhs, double rhs)
        {
            var v = new double[lhs.Count];
            for (int i = 0; i < lhs.vector.Length; ++i)
                v[i] = lhs.vector[i] - rhs;
            return new Derivatives(v);
        }

        /// <summary>
        /// Multiply derivatives by a number.
        /// </summary>
        public static Derivatives operator *(Derivatives lhs, double rhs)
        {
            var v = new double[lhs.Count];
            for (int i = 0; i < lhs.vector.Length; ++i)
                v[i] = lhs.vector[i] * rhs;
            return new Derivatives(v);
        }

        /// <summary>
        /// Divide derivatives by a number.
        /// </summary>
        public static Derivatives operator /(Derivatives lhs, double rhs)
        {
            var v = new double[lhs.Count];
            for (int i = 0; i < lhs.vector.Length; ++i)
                v[i] = lhs.vector[i] / rhs;
            return new Derivatives(v);
        }

        /// <summary>
        /// Add a number to derivatives.
        /// </summary>
        public static Derivatives operator +(double lhs, Derivatives rhs) =>
            rhs + lhs;

        /// <summary>
        /// Multiply derivatives by a number.
        /// </summary>
        public static Derivatives operator *(double lhs, Derivatives rhs) =>
            rhs * lhs;
    }
}
