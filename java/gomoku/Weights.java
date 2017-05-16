package gomoku;


/** Helper class for MinimaxAgent
 * @author TakLee96 */
class Weights {

    static Counter blackEval = new Counter();
    static {
        blackEval.put("-oox"  ,  10       );
        blackEval.put("-o-ox" ,  10       );
        blackEval.put("-xxo"  , -1        );
        blackEval.put("-xxo"  , -1        );
        blackEval.put("-x-xo" , -1        );
        blackEval.put("-o-o-" ,  1000     );
        blackEval.put("-oo-"  ,  1000     );
        blackEval.put("-o-oox",  1000     );
        blackEval.put("-ooox" ,  1000     );
        blackEval.put("o-o-o" ,  1000     );
        blackEval.put("-x-x-" , -100      );
        blackEval.put("-xx-"  , -100      );
        blackEval.put("-x-xxo", -100      );
        blackEval.put("-xxxo" , -100      );
        blackEval.put("x-x-x" , -100      );
        blackEval.put("-o-oo-",  100000   );
        blackEval.put("-ooo-" ,  100000   );
        blackEval.put("-x-xx-", -10000    );
        blackEval.put("-xxx-" , -10000    );
        blackEval.put("-oooo-",  10000000 );
        blackEval.put("-oooox",  10000000 );
        blackEval.put("four-o",  10000000 );
        blackEval.put("-xxxx-", -1000000  );
        blackEval.put("-xxxxo", -1000000  );
        blackEval.put("four-x", -1000000  );
        blackEval.put("win-o" ,  10000000 );
    }

    static Counter whiteEval = new Counter();
    static {
        whiteEval.put("-xxo"  , -10       );
        whiteEval.put("-x-xo" , -10       );
        whiteEval.put("-oox"  ,  1        );
        whiteEval.put("-o-ox" ,  1        );
        whiteEval.put("-x-x-" , -1000     );
        whiteEval.put("-xx-"  , -1000     );
        whiteEval.put("-x-xxo", -1000     );
        whiteEval.put("-xxxo" , -1000     );
        whiteEval.put("x-x-x" , -1000     );
        whiteEval.put("-o-o-" ,  100      );
        whiteEval.put("-oo-"  ,  100      );
        whiteEval.put("-o-oox",  100      );
        whiteEval.put("-ooox" ,  100      );
        whiteEval.put("o-o-o" ,  100      );
        whiteEval.put("-x-xx-", -100000   );
        whiteEval.put("-xxx-" , -100000   );
        whiteEval.put("-o-oo-",  10000    );
        whiteEval.put("-ooo-" ,  10000    );
        whiteEval.put("-xxxx-", -10000000 );
        whiteEval.put("-xxxxo", -10000000 );
        whiteEval.put("four-x", -10000000 );
        whiteEval.put("-oooo-",  1000000  );
        whiteEval.put("-oooox",  1000000  );
        whiteEval.put("four-o",  1000000  );
        whiteEval.put("win-x" , -10000000 );
    }

}