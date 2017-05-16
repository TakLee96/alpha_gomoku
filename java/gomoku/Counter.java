package gomoku;

import java.util.Set;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Collections;
import java.io.BufferedReader;
import java.io.FileReader;

/** Counter, just a map that return 0.0 for none-existing key
 * @author TakLee96 */
class Counter {

    private HashMap<String, Number> map;
    public Counter() {
        map = new HashMap<String, Number>();
    }

    public void put(String key, Number val) {
        if (key != null)
            if (val.doubleValue() != 0.0)
                map.put(key, val);
            else
                map.remove(key);
    }

    public int getInt(String key) {
        return get(key).intValue();
    }

    public double getDouble(String key) {
        return get(key).doubleValue();
    }

    private Number get(String key) {
        if (key == null)
            return 0.0;
        Number result = map.get(key);
        if (result == null)
            return 0.0;
        return result;
    }

    public boolean containsKey(String key) {
        return map.containsKey(key);
    }

    public Set<String> keySet() {
        return map.keySet();
    }

    public void add(Counter diff) {
        int count = 0;
        for (String key : diff.keySet()) {
            count = get(key).intValue() + diff.get(key).intValue();
            if (count < 0) {
                throw new RuntimeException("negative count for " + key);
            }
            put(key, count);
        }
    }

    public void sub(Counter diff) {
        int count = 0;
        for (String key : diff.keySet()) {
            count = get(key).intValue() - diff.get(key).intValue();
            if (count < 0) {
                throw new RuntimeException("negative count for " + key);
            }
            put(key, count);
        }
    }

    public double mul(Counter features) {
        double total = 0.0;
        for (String key : features.keySet())
            total += get(key).doubleValue() * features.get(key).intValue();
        return total;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("{");

        ArrayList<String> sortedKeys = new ArrayList<String>(map.keySet());
        Collections.sort(sortedKeys);
        for (String key : sortedKeys) {
            if (map.get(key).doubleValue() != 0.0)
                sb.append(" " + key + " : " + map.get(key) + ",");
        }
        sb.append(" }");
        return sb.toString();
    }

}
