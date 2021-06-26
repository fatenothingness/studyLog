package handle;

import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;
import org.aspectj.lang.annotation.Pointcut;
import org.springframework.stereotype.Component;

@Aspect
@Component
public class TestLogAspect {

    @Pointcut("@annotation(handle.TestLog)")
    private void pointCut(){

    }

    @Before("pointCut()")
    public void doBefore(JoinPoint joinPoint){
        String name = joinPoint.getSignature().getName();
        String simpleName = joinPoint.getSignature().getDeclaringType().getSimpleName();
        System.out.println(name+simpleName+"准备开始运行");
    }
    @Around("pointCut()")
    public Object doAround(ProceedingJoinPoint joinPoint) throws Throwable {
        System.out.println(joinPoint.getSignature().getName()+"运行ing");
        Object res = null;
        Object[] args = joinPoint.getArgs();
        if (args[0] instanceof Integer){
            args[0] = (Integer)args[0]+1;
        }
        try {
            res = joinPoint.proceed();
        }catch (Throwable t){
            throw t;
        }
        return res;
    }
}
