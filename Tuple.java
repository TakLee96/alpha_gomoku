public class Tuple<T> {
    public T x;
    public T y;
    public Tuple(T x, T y) {
        this.x = x;
        this.y = y;
    }

    @Override
    public boolean equals(Object other) {
        Tuple<T> o = (Tuple<T>) other;
        return o.x == x && o.y == y;
    }

    @Override
    public String toString() {
        return "(" + x + "," + y + ")";
    }

    @Override
    public int hashCode() {
        return x * 1009 + y;
    }
}