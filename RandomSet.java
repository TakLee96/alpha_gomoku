import java.util.Iterator;
import java.util.Random;
import java.util.HashSet;
import java.util.ArrayList;

public class RandomSet<T> implements Iterable<T> {

    private HashSet<T> delegate;
    private ArrayList<T> store;
    private Random random;
    private boolean cacheGood;
    public RandomSet() {
        delegate = new HashSet<T>();
        store = new ArrayList<T>();
        random = new Random(System.nanoTime());
        cacheGood = true;
    }

    public boolean contains(T elem) {
        return delegate.contains(elem);
    }

    private void build() {
        cacheGood = true;
        store = new ArrayList<T>(delegate);
    }

    public void add(T elem) {
        if (!delegate.contains(elem)) {
            delegate.add(elem);
            store.add(elem);
        }
    }

    public T pollRandom() {
        if (!cacheGood) build();
        return store.get(random.nextInt(store.size()));
    }

    public int size() {
        return delegate.size();
    }

    public boolean isEmpty() {
        return delegate.isEmpty();
    }

    public void remove(T elem) {
        cacheGood = false;
        delegate.remove(elem);
    }

    @Override
    public Iterator<T> iterator() {
        return (new ArrayList<T>(delegate)).iterator();
    }

}
