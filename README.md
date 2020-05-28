# AutoDiffSharp

Simple automatic differentiation (AD) in C# that uses operator overloading to implement dual Duals.

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

Each parameter carries an extra 'ϵ' value corresponding to the derivative, and this extra value is distinct from the 'ϵ' values of all other parameters. In order to carry all the derivatives of all input parameters, we'd need an array of derivatives where the index corresponds to the derivative for the parameter at that index. This is pretty inefficient however, but if you're interested to see how that works, check out the [history of this repo here](https://github.com/naasking/AutoDiffSharp/tree/d5fd521cf784feab7e7209dd078abe9a7ff2f4be).

In order to make differentiation efficient, we'll first implement so-called forward mode automatic differentiation with dual numbers:

```csharp
public readonly struct Dual
{
    public readonly double Magnitude;
    public readonly double Derivative;

    internal Dual(double m, double d)
    {
        this.Magnitude = m;
        this.Derivative = d;
    }
}
```

Since the dual number only carries around a single derivative, it must correspond to only one of the ϵ<sub>x</sub> values, which means we can only take the derivative of one of the input parameters at a time. See the section below on reverse mode automatic differentiation for an efficient way to compute *all* derivatives simultaneously.

A differentiable function of 3 parameters would have this signature:

```csharp
Dual Function(Dual x0, Dual x1, Dual x2)
```

Internally, differentiation invokes the function like this:

```csharp
public static Dual DifferentiateX0At(
    double x0, double x1, double x2,
    Func<Dual, Dual, Dual, Dual> func) =>
    func(new Dual(x0, 1),
         new Dual(x1, 0),
         new Dual(x2, 0));
```

Each parameter is initialized with zeroes except for the derivative we're interested in.

Operators are now pretty straightforward, corresponding to operations on the magnitude and derivative:

```csharp
public static Dual operator +(Dual lhs, Dual rhs) =>
    new Dual(lhs.Magnitude + rhs.Magnitude,
             lhs.Derivative + rhs.Derivative);

public static Dual operator *(Dual lhs, Dual rhs) =>
    new Dual(lhs.Magnitude * rhs.Magnitude,
             lhs.Derivative * rhs.Magnitude + rhs.Derivative * lhs.Magnitude);
```

Once you have your return value of type `Dual`, you can access the derivative:

```csharp
var y_dx0 = Calculus.DifferentiateX0At(x0, x1, function);
Console.WriteLine("x0' = " + y_dx0.Derivative);

var y_dx1 = Calculus.DifferentiateX1At(x0, x1, function);
Console.WriteLine("x1' = " + y_dx1.Derivative);
```

# Reverse Mode Automatic Differentiation

The above description is for forward-mode AD, but there's a dual representation of forward mode with properties that can replace the vector representation above with an abstraction that takes only linear space. This is called reverse mode automatic differentiation. Instead of computing the derivative alongside the value, we instead construct a *continuation* that computes the derivatives *backwards*, see `Codual.cs`:

```csharp
public readonly struct Codual
{
    public readonly double Magnitude;
    public readonly Action<double> Derivative;

    internal Dual(double m, Action<double> d)
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

The advantage here is that the continuation eliminates the need to build NxM arrays to track the derivative vectors as I was doing in an older version of this repo, while still retaining the ability to compute the derivatives of all input parameters simultaneously.

In general, forward-mode AD is best suited for functions of type R->R<sup>N</sup>, which are functions of a single real Dual to a set of real Duals, where reverse mode AD is best suited for functions of type R<sup>N</sup>->R. The latter type are pretty common in machine learning these days.

This means reverse mode AD is not suitable for functions that outputs many results, or functions in which you're only interested in a small subset of the derivatives.
