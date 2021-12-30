package handle;

public class Teacher {

    public  static  int value =123;

    static {
        value = 456;
        System.out.println("super doing"+value);
    }
    public static  int v2 = value;
}
