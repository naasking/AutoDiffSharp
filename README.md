# AutoDiffSharp

Simple automatic differentiation (AD) in C# that uses operator overloading to implement dual numbers.

This isn't a super efficient implementation, but it's probably fine for small tests and for learning how AD works.

If you only need AD for differentiating with respect to one variable, see the "Optimizations" section below for a very efficient specialization of the general approach described here.

# Dual Numbers

The most straightforward implemenation of AD is based on [dual numbers](https://en.wikipedia.org/wiki/Automatic_differentiation#Automatic_differentiation_using_dual_numbers). Each regular number is augmented with an extra term corresponding to it's derivative:

    real number x   =(dual number)=>   x + x'*ϵ

Arithmetic and other mathematical functions then have translations to operating on these extended number types as follows:

|Operator|Translated|
|--------|----------|
|<x, x'> + <y, y'>|<x + y, x' + y'>
|<x, x'> - <y, y'>|<x - y, x' - y'>
|<x, x'> \* <y, y'>|<x\*y, y'\*x - y*x'>
|<x, x'> / <y, y'>|<x / y, (x'\*y - x\*y')/y<sup>2</sup>>
|<x, x'><sup>k</sup>|<x<sup>k</sup>, k\*x<sup>(k-1)</sup>\*x'>

See the wikipedia page for more transformations, like standard trig functions.

# Dual Numbers from Operator Overloading

Invoking a function "f" with dual numbers operates like this, in math notation:

> f(x0 + ϵ<sub>x1</sub>, x1 + ϵ<sub>x2</sub>, x2 + ϵ<sub>x2</sub>)

Each parameter carries an extra 'ϵ' value corresponding to the derivative, and this extra value is distinct from the 'ϵ' values of all other parameters. However, as you can see in the translation table, these derivatives interact with one another in some operators, so a general number type carries a vector for the coefficients of all other parameters. Here's the basic number type:

```csharp
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
```

A differentiable function of 3 parameters would have this signature:

```csharp
Number Function(Number x0, Number x1, Number x2)
```

Internally, differentiation invokes the function like this:

```csharp
public static Number DifferentiateAt(
    double x0, double x1, double x2,
    Func<Number, Number, Number, Number> func) =>
    func(new Number(x0, 1, 0, 0),
         new Number(x1, 0, 1, 0),
         new Number(x2, 0, 0, 1));
```

Each parameter is initialized with zeroes everywhere except for a 1 in the derivative slot corresponding to its position in the argument list. This is the necessary starting configuration for automatic differentiation in order to compute the derivatives for any of the parameters.

Operators are now pretty straightforward, corresponding to operations on the magnitude and pairwise operations on each index of the array:

```csharp
public static Number operator +(Number lhs, Number rhs) =>
    new Number(lhs.Magnitude + rhs.Magnitude,
               lhs.Derivatives + rhs.Derivatives);

public static Number operator *(Number lhs, Number rhs) =>
    new Number(lhs.Magnitude * rhs.Magnitude,
               lhs.Derivatives * rhs.Magnitude + rhs.Derivatives * lhs.Magnitude);
```

Obviously you can't add or multiply two `double[]` like I've shown here, but the actual implementation hides the array behind a `Derivatives` type that exposes the arithmetic operators:

```csharp
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
```

Once you have your return value of type `Number`, you can access the derivatives of each parameter by its index:

```csharp
var y = Calculus.DifferentiateAt(x0, x1, function);
Console.WriteLine("x0' = " + y.Derivatives[0]);
Console.WriteLine("x1' = " + y.Derivatives[1]);
```

# Optimizations

Most presentations of automatic differentiation show examples where you differentiate a function with respect to only a single parameter, but this technique computes *every derivative simultaneously*. Obviously that's more general, but you typically don't need all of the derivatives which makes this technique a little wasteful.

So as a first optimization, start with the starting configuration for automatic differentiation described above and consider what happens when you're interested in the derivative of x0 *only*:

```csharp
public static Number Differentiate_X0(
    double x0, double x1, double x2,
    Func<Number, Number, Number, Number> func) =>
    func(new Number(x0, 1, 0, 0),
         new Number(x1, 0, 0, 0),
         new Number(x2, 0, 0, 0));
```

When you only want one of the derivatives, the ϵ coefficient of all other parameters would be zero, and all of those array slots filled with zeroes would stay zero throughout the whole computation. So toss them out!

Create a specialized `Number` type that doesn't incur any array allocations at all by replacing `Derivatives` with a single `System.Double` corresponding to the one parameter that's being differentiated. That parameter gets a 1 as the extra term when differentiating, the rest all start with 0. See `Dual.cs` for an implementation of this type:

```csharp
public readonly struct Dual
{
    public readonly double Magnitude;
    public readonly double Derivative;

    internal Number(double m, double d)
    {
        this.Magnitude = m;
        this.Derivative = d;
    }
}
```

So while you can only differentiate with respect to one variable at a time with `Dual`, you only need to carry around an extra double for each step in the calculation. This would be very efficient!

# Reverse Mode Automatic Differentiation

The above description is for forward-mode AD, but there's a dual representation of forward mode with properties that can replace the vector representation above with an abstraction that takes only linear space. This is called reverse mode automatic differentiation. Instead of computing the derivative alongside the value, we instead construct a *continuation* that computes the derivatives *backwards*, see `Codual.cs`:

```csharp
public readonly struct Codual
{
    public readonly double Magnitude;
    public readonly Action<double> Derivative;

    internal Number(double m, Action<double> d)
    {
        this.Magnitude = m;
        this.Derivative = d;
    }
    
    public static Codual operator +(Codual lhs, Codual rhs) =>
        new Codual(lhs.Magnitude + rhs.Magnitude, dx =>
        {
            lhs.Derivative(dx);
            rhs.Derivative(dx);
        });
        
    public static Codual operator *(Codual lhs, Codual rhs) =>
        new Codual(lhs.Magnitude * rhs.Magnitude, dx =>
        {
            lhs.Derivative(dx * rhs.Magnitude);
            rhs.Derivative(dx * lhs.Magnitude);
        });
}
```

The advantage here is that the continuation eliminates the need to build NxM arrays to track the derivative vectors, while still retaining the ability to compute the derivatives of all input parameters simultaneously.


In general, forward-mode AD is best suited for functions of type R->R<sup>N</sup>, which are functions of a single real number to a set of real numbers, where reverse mode AD is best suited for functions of type R<sup>N</sup>->R. The latter type are pretty common in machine learning these days.