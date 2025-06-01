const std = @import("std");
const testing = std.testing;

const Op = enum {
    init,
    add,
    mul,
    tanh,
};

const Value = struct {
    data: f64,
    grad: f64 = 0.0,
    prev: [2]?*Value = .{ null, null },
    op: Op = Op.init,
    label: []const u8 = "",
    backward: *const fn (self: *Value) void = dummy_backward,

    fn dummy_backward(self: *Value) void {
        _ = &self;
    }

    fn init(data: f64, label: []const u8) Value {
        return Value{
            .data = data,
            .label = label,
        };
    }

    fn add(self: *Value, other: *Value) Value {
        return Value{
            .data = self.data + other.data,
            .prev = .{ self, other },
            .op = Op.add,
            .backward = &add_backward,
        };
    }

    fn add_backward(self: *Value) void {
        self.prev[0].?.grad = 1.0 * self.grad;
        self.prev[1].?.grad = 1.0 * self.grad;
    }

    fn mul(self: *Value, other: *Value) Value {
        return Value{
            .data = self.data * other.data,
            .prev = .{ self, other },
            .op = Op.mul,
            .backward = mul_backward,
        };
    }

    fn mul_backward(self: *Value) void {
        self.prev[0].?.grad = self.prev[1].?.data * self.grad;
        self.prev[1].?.grad = self.prev[0].?.data * self.grad;
    }

    fn tanh(self: *Value) Value {
        const x = self.data;
        const t = (std.math.exp(2 * x) - 1) / (std.math.exp(2 * x) + 1);
        return Value{
            .data = t,
            .prev = .{ self, null },
            .op = Op.tanh,
            .backward = tanh_backward,
        };
    }

    fn tanh_backward(self: *Value) void {
        const t = self.data;
        self.prev[0].?.grad = (1 - t * t) * self.grad;
    }

    fn showGraph(self: *const Value, indent: usize) void {
        std.debug.print("{s: >[1]}Value(label: {[2]s}, op: {[3]s}, data: {[4]d}, grad: {[5]d})\n", .{ "", indent, self.label, @tagName(self.op), self.data, self.grad });

        for (self.prev) |p| {
            if (p) |v| {
                v.showGraph(indent + 2);
            }
        }
    }
};

test "testing Value" {
    var x1 = Value.init(2.0, "x1");
    var x2 = Value.init(0.0, "x2");
    var w1 = Value.init(-3.0, "w1");
    var w2 = Value.init(1.0, "w2");
    var b = Value.init(6.8813735870195432, "b");

    var x1w1 = x1.mul(&w1);
    x1w1.label = "x1*w1";
    var x2w2 = x2.mul(&w2);
    x2w2.label = "x2*w2";
    var x1w1x2w2 = x1w1.add(&x2w2);
    x1w1x2w2.label = "x1w1 + x2w2";
    var n = x1w1x2w2.add(&b);
    n.label = "n";
    var o = n.tanh();
    o.label = "o";

    o.grad = 1.0;
    o.backward(&o);
    n.backward(&n);
    b.backward(&b);
    x1w1x2w2.backward(&x1w1x2w2);
    x1w1.backward(&x1w1);
    x2w2.backward(&x2w2);

    o.showGraph(1);
}
