import interactive_host
import sys


def main(args):
    class Advisor:
        to_return = False
        counter = 0

        def advice(self, _):
            l = [0.5 for _ in range(1 + 32 - 2)]
            l[int(args[0])] = float(args[1])
            return l

        def before_advice(self, tc, fc):
            json = fc.readline()
            print(json)
            heuristic = int.from_bytes(fc.read(8))
            print(heuristic)
            fc.readline()

        def after_advice(self, tc, fc):
            json = fc.readline()
            print(json)
            action = bool(int.from_bytes(fc.read(1)))
            print(action)
            fc.readline()
            if args[2] == 'instrument':
                tc.write(bytes([1]))
                begin = ("test_loop_begin_" + str(self.counter)).encode('ascii') + bytes([0])
                end = ("test_loop_end_" + str(self.counter)).encode('ascii') + bytes([0])
                self.counter += 1
                tc.write(begin)
                tc.write(end)
                tc.flush()
            else:
                # Respond that we do not want instrumentation
                tc.write(bytes([0]))
                tc.flush()

    a = Advisor()
    interactive_host.run_interactive(args[3],
                                     a.advice,
                                     args[4:],
                                     a.before_advice,
                                     a.after_advice)


if __name__ == "__main__":
    main(sys.argv[1:])
