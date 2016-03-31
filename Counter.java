import java.util.HashMap;

public class Counter {

    private HashMap<String, Double> map;
    public Counter() {
        map = new HashMap<String, Double>();
    }

    public void put(String key, double val) {
        map.put(key, val);
    }

    public double get(String key) {
        Double result = map.get(key);
        if (result == null) {
            return 0.0;
        }
        return result;
    }

}