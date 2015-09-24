package utils;


public class IntDoubleTuple {
	private final int _id;
	private final double _value;
	
	public IntDoubleTuple(int id, double value){
		_id = id;
		_value = value;
	}
	
	public int getId(){
		return _id;
	}
	
	public double getValue(){
		return _value;
	}
}
