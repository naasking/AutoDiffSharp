# AutoDiffSharp

Simple automatic differentiation (AD) in C# that uses operator overloading
to implement dual numbers.

This isn't a super efficient implementation, but it's probably fine for
small functions and tests, and for learning how AD works.

# Dual Numbers

The most straightforward implemenation of AD is based on [dual numbers](https://en.wikipedia.org/wiki/Automatic_differentiation#Automatic_differentiation_using_dual_numbers). Each
regular number is augmented with an extra term corresponding to it's derivative:

    real number x   =(dual number)=>   x + x'*ϵ

Arithmetic and other mathematical functions then have translations to operating
on these extended number types as follows:

|Operator|Translated|
|--------|----------|
|<x, x'> + <y, y'>|<x + y, x' + y'>
|<x, x'> - <y, y'>|<x - y, x' - y'>
|<x, x'> \* <y, y'>|<x\*y, y'\*x - y*x'>
|<x, x'> / <y, y'>|<x / y, (x'\*y - x\*y')/y^2>
|<x, x'><sup>k</sup>|<x<sup>k</sup>, k\*x<sup>(k-1)</sup>\*x'>

See the wikipedia page for more transformations, like standard trig functions.

# Core

Invoking a function "f" with dual numbers operates like this, in math notation:

> f(x0 + ϵ<sub>x1</sub>, x1 + ϵ<sub>x2</sub>, x2 + ϵ<sub>x2</sub>)

So each parameter gets its own differentiable extra parameter, distinct from all
others. However, as you can see in the translation table, these all interact with
one another in some operators, so each function parameter has to carry a vector
corresponding to the coefficients of all other parameters:

    public readonly struct Number
    {
        public readonly double Magnitude;
        public readonly double[] Derivatives;

        internal Number(double m, params double[] d)
        {
            this.Magnitude = m;
            this.Derivatives = d;
        }
    }

A differentiable function of 3 parameters would have this signature:

    Number Function(Number x0, Number x1, Number x2)

Internally, differentiation invokes the function like this:

    public static Number DifferentiateAt(
        double x0, double x1, double x2, Func<Number, Number, Number, Number> func) =>
        func(new Number(x0, 1, 0, 0), new Number(x1, 0, 0, 1), new Number(x2, 0, 0, 1));

As you can see, each parameter has a 1 in the derivative slot corresponding to
its position in the argument list, and zeroes everywhere else.

Operators are now pretty straightforward, corresponding to operations
on the magnitude and pairwise operations on each index of the array:

    public static Number operator +(Number lhs, Number rhs) =>
        new Number(lhs.Magnitude + rhs.Magnitude,
                   lhs.Derivatives + rhs.Derivatives);

    public static Number operator *(Number lhs, Number rhs) =>
        new Number(lhs.Magnitude * rhs.Magnitude,
                   lhs.Derivatives * rhs.Magnitude + rhs.Derivatives * lhs.Magnitude);

Obviously you can't add or multiply two `double[]` like I've shown here,
but the actual implementation hides the array behind a `Derivatives` type
that exposes the arithmetic operators:

    public static Derivatives operator +(Derivatives lhs, Derivatives rhs)
    {
        var v = new double[lhs.Count];
        for (int i = 0; i < lhs.vector.Length; ++i)
            v[i] = lhs.vector[i] + rhs.vector[i];
        return new Derivatives(v);
    }
    
    public static Derivatives operator *(Derivatives lhs, Derivatives rhs)
    {
        var v = new double[lhs.Count];
        for (int i = 0; i < lhs.vector.Length; ++i)
            v[i] = lhs.vector[i] * rhs.vector[i];
        return new Derivatives(v);
    }

Once you have your return value of type `Number`, you can access the derivatives
of each parameter by its index:

    var y = Calculus.DifferentiateAt(x0, x1, f);
    Console.WriteLine("x0' = " + y.Derivatives[0]);
    Console.WriteLine("x1' = " + y.Derivatives[1]);
