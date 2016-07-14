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

    private static String weights = String.join("\n", "{",
        "white : -xxo   : -10       ",
        "white : -x-xo  : -10       ",
        "white : -oox   :  1        ",
        "white : -o-ox  :  1        ",
        "white : -x-x-  : -1000     ",
        "white : -xx-   : -1000     ",
        "white : -x-xxo : -1000     ",
        "white : -xxxo  : -1000     ",
        "white : x-x-x  : -1000     ",
        "white : -o-o-  :  100      ",
        "white : -oo-   :  100      ",
        "white : -o-oox :  100      ",
        "white : -ooox  :  100      ",
        "white : o-o-o  :  100      ",
        "white : -x-xx- : -100000   ",
        "white : -xxx-  : -100000   ",
        "white : -o-oo- :  10000    ",
        "white : -ooo-  :  10000    ",
        "white : -xxxx- : -10000000 ",
        "white : -xxxxo : -10000000 ",
        "white : four-x : -10000000 ",
        "white : -oooo- :  1000000  ",
        "white : -oooox :  1000000  ",
        "white : four-o :  1000000  ",
        "white : win-x  : -100000000",
        "black : -oox   :  10       ",
        "black : -o-ox  :  10       ",
        "black : -xxo   : -1        ",
        "black : -x-xo  : -1        ",
        "black : -o-o-  :  1000     ",
        "black : -oo-   :  1000     ",
        "black : -o-oox :  1000     ",
        "black : -ooox  :  1000     ",
        "black : o-o-o  :  1000     ",
        "black : -x-x-  : -100      ",
        "black : -xx-   : -100      ",
        "black : -x-xxo : -100      ",
        "black : -xxxo  : -100      ",
        "black : x-x-x  : -100      ",
        "black : -o-oo- :  100000   ",
        "black : -ooo-  :  100000   ",
        "black : -x-xx- : -10000    ",
        "black : -xxx-  : -10000    ",
        "black : -oooo- :  10000000 ",
        "black : -oooox :  10000000 ",
        "black : four-o :  10000000 ",
        "black : -xxxx- : -1000000  ",
        "black : -xxxxo : -1000000  ",
        "black : four-x : -1000000  ",
        "black : win-o  :  100000000",
    "}");

    private static DecimalFormat formatter = new DecimalFormat("#0.000");

    private HashMap<String, Number> map;
    public Counter() {
        map = new HashMap<String, Number>();
    }

    public static void read(Counter black, Counter white) {
          for (String line : weights.split("")) {
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
