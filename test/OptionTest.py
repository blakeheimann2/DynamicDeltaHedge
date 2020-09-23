import unittest
from app.Securities import Option



class OptionTest(unittest.TestCase):
    def testATM(self):
        put = Option(100, 100, 5/365, 0.05, is_call=False)
        print(put.getInfo())
        print(put.getPrice(0.21))
        print(put.getImpliedVol(1))

        call = Option(100, 100, 5/365, 0.05, is_call=True)
        print(call.getInfo())
        print(call.getPrice(0.21))
        print(call.getImpliedVol(1))

    def testITM(self):
        put = Option(95, 100, 30/365, 0.05, is_call=False)
        print(put.getInfo())
        print(put.getPrice(0.25))
        print(put.getImpliedVol(5.6577))

        call = Option(105, 100, 30/365, 0.05, is_call=True)
        print(call.getInfo())
        print(call.getPrice(0.25))
        print(call.getImpliedVol(6.391))

    def testOTM(self):
        put = Option(105, 100, 1/365, 0.05, is_call=False)
        print(put.getInfo())
        print(put.getPrice(0.25))
        print(put.getImpliedVol(0))

        call = Option(95, 100, 1/365, 0.05, is_call=True)
        print(call.getInfo())
        print(call.getPrice(0.25))
        print(call.getImpliedVol(0))

    def testDelta(self):
        put = Option(100, 50, 1, 0.05, is_call=False)
        print(put.getInfo())
        print(put.getPrice(0.25))
        print(put.getGreeks(.25))

    def testVega(self):
        put = Option(50, 100, 1, 0.05, is_call=False)
        print(put.getInfo())
        print(put.getPrice(0.25))
        print(put.getGreeks(.25))

    def testGamma(self):
        put = Option(110, 100, 1, 0.05, is_call=False)
        print(put.getInfo())
        print(put.getPrice(0.25))
        print(put.getGreeks(.25))

    def testTheta(self):
        put = Option(110, 100, 2, 0.05, is_call=False)
        print(put.getInfo())
        print(put.getPrice(0.25))
        print(put.getGreeks(.25))

if __name__ == '__main__':
    unittest.main()
