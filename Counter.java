import java.util.HashMap;
import java.text.DecimalFormat;

public class Counter {

    private static DecimalFormat formatter = new DecimalFormat("#0.000E0");

    private HashMap<String, Double> map;
    public Counter() {
        map = new HashMap<String, Double>();
    }

    public Counter(Counter other) {
        this();
        map.putAll(other.map);
    }

    public void put(String key, double val) {
        if (val != 0.0)
            map.put(key, val);
    }

    public double get(String key) {
        Double result = map.get(key);
        if (result == null)
            return 0.0;
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        for (String key : map.keySet()) {
            sb.append("    " + key + " : " + formatter.format(map.get(key)) + "\n");
        }
        sb.append("}");
        return sb.toString();

    }

}
