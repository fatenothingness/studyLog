package handle;

import org.springframework.stereotype.Component;

@Component
public class Student implements Person {

    private String name;

    private Integer age;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Integer getAge() {
        return age;
    }

    public void setAge(Integer age) {
        this.age = age;
    }

    @Override
    public void sayHello() {
        System.out.println(name+"今年"+age+"岁");
    }

    @TestLog
    public void getInfo(Integer num) {
        System.out.println(name+"今年"+num+"岁");
    }
}
