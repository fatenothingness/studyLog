package handle;



import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

public class TestHandle<T> implements InvocationHandler {

     T target;

     public TestHandle(T target){
         this.target = target;
     }

    @Override
    public Object invoke(Object o, Method method, Object[] objects) throws Throwable {
        System.out.println("代理类执行方法:"+ method.getName());
        Object res = method.invoke(target,objects);
        return res;
    }
}
