package utils;

import java.util.Comparator;

public class IntDoubleTupleComperator implements Comparator<IntDoubleTuple> {

	@Override
	public int compare(IntDoubleTuple o1, IntDoubleTuple o2) {
		double v1 = o1.getValue();
		double v2 = o2.getValue();
		
		if(v1 < v2){
			return 1;
		}else if(v2 < v1){
			return -1;
		}else{
			return 0;
		}
	}
}
