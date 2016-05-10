package gomoku;

import java.util.Set;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Collections;
import java.text.DecimalFormat;
import java.io.BufferedReader;
import java.io.FileReader;

/** Counter, just a map that return 0.0 for none-existing key
 * @author TakLee96 */
public class Counter {

    private static DecimalFormat formatter = new DecimalFormat("#0.000");

    private HashMap<String, Number> map;
    public Counter() {
        map = new HashMap<String, Number>();
    }

    public static void read(Counter black, Counter white, String file) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            for (String line = br.readLine(); line != null; line = br.readLine()) {
                String[] parts = line.split(":");
                if (parts.length == 3)
                    if (parts[0].trim().equals("black"))
                        black.put(parts[1].trim(), Double.parseDouble(parts[2].trim()));
                    else
                        white.put(parts[1].trim(), Double.parseDouble(parts[2].trim()));
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void put(String key, Number val) {
        if (key != null) map.put(key, val);
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
                System.out.println(key);
                throw new RuntimeException("negative count");
            }
            put(key, count);
        }
    }

    public void sub(Counter diff) {
        int count = 0;
        for (String key : diff.keySet()) {
            count = get(key).intValue() - diff.get(key).intValue();
            if (count < 0) {
                System.out.println(key);
                throw new RuntimeException("negative count");
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
