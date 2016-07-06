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

    private static String weights = "{\n"+
        "white : -xxo   : -10\n"+
        "white : -x-xo  : -10\n"+
        "white : -oox   :  1\n"+
        "white : -o-ox  :  1\n"+
        "white : -x-x-  : -1000\n"+
        "white : -xx-   : -1000\n"+
        "white : -x-xxo : -1000\n"+
        "white : -xxxo  : -1000\n"+
        "white : x-x-x  : -1000\n"+
        "white : -o-o-  :  100\n"+
        "white : -oo-   :  100\n"+
        "white : -o-oox :  100\n"+
        "white : -ooox  :  100\n"+
        "white : o-o-o  :  100\n"+
        "white : -x-xx- : -100000\n"+
        "white : -xxx-  : -100000\n"+
        "white : -o-oo- :  10000\n"+
        "white : -ooo-  :  10000\n"+
        "white : -xxxx- : -10000000\n"+
        "white : -xxxxo : -10000000\n"+
        "white : four-x : -10000000\n"+
        "white : -oooo- :  1000000\n"+
        "white : -oooox :  1000000\n"+
        "white : four-o :  1000000\n"+
        "white : win-x  : -100000000\n"+
        "black : -oox   :  10\n"+
        "black : -o-ox  :  10\n"+
        "black : -xxo   : -1\n"+
        "black : -x-xo  : -1\n"+
        "black : -o-o-  :  1000\n"+
        "black : -oo-   :  1000\n"+
        "black : -o-oox :  1000\n"+
        "black : -ooox  :  1000\n"+
        "black : o-o-o  :  1000\n"+
        "black : -x-x-  : -100\n"+
        "black : -xx-   : -100\n"+
        "black : -x-xxo : -100\n"+
        "black : -xxxo  : -100\n"+
        "black : x-x-x  : -100\n"+
        "black : -o-oo- :  100000\n"+
        "black : -ooo-  :  100000\n"+
        "black : -x-xx- : -10000\n"+
        "black : -xxx-  : -10000\n"+
        "black : -oooo- :  10000000\n"+
        "black : -oooox :  10000000\n"+
        "black : four-o :  10000000\n"+
        "black : -xxxx- : -1000000\n"+
        "black : -xxxxo : -1000000\n"+
        "black : four-x : -1000000\n"+
        "black : win-o  :  100000000\n"+
    "}";

    private static DecimalFormat formatter = new DecimalFormat("#0.000");

    private HashMap<String, Number> map;
    public Counter() {
        map = new HashMap<String, Number>();
    }

    public static void read(Counter black, Counter white) {
          for (String line : weights.split("\n")) {
              String[] parts = line.split(":");
              if (parts.length == 3)
                  if (parts[0].trim().equals("black"))
                      black.put(parts[1].trim(), Double.parseDouble(parts[2].trim()));
                  else
                      white.put(parts[1].trim(), Double.parseDouble(parts[2].trim()));
          }
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
