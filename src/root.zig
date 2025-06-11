// DONE TILL 1:44:49 https://youtu.be/VMj-3S1tku0?t=6289
const std = @import("std");
const expectApproxEqAbs = std.testing.expectApproxEqAbs;

const Op = enum {
    init,
    add,
    mul,
    // div, // TODO atm div is a * b ** -1
    tanh,
    exp,
    pow,
};

fn build_topo(
    allocator: std.mem.Allocator,
    node: *Value,
    visited: *std.AutoHashMap(*Value, bool),
    topo: *std.ArrayList(*Value),
) !void {
    if (!visited.contains(node)) {
        try visited.put(node, true);

        for (node.prev) |p| {
            if (p) |child| {
                try build_topo(allocator, child, visited, topo);
            }
        }
        try topo.append(node);
    }
}

// TODO ptr situation
pub const Value = struct {
    data: f64,
    grad: f64 = 0.0,
    prev: [2]?*Value = .{ null, null },
    op: Op = Op.init,
    label: []const u8 = "",
    backward: *const fn (self: *Value) void = dummy_backward,

    fn dummy_backward(self: *Value) void {
        _ = &self;
    }

    pub fn init(data: f64, label: []const u8) Value {
        return Value{
            .data = data,
            .label = label,
        };
    }

    pub fn add(self: *Value, other: *Value) Value {
        // TODO accept Value or float.
        return Value{
            .data = self.data + other.data,
            .prev = .{ self, other },
            .op = Op.add,
            .backward = &add_backward,
        };
    }

    pub fn add_backward(self: *Value) void {
        self.prev[0].?.grad += 1.0 * self.grad;
        self.prev[1].?.grad += 1.0 * self.grad;
    }

    pub fn mul(self: *Value, other: *Value) Value {
        return Value{
            .data = self.data * other.data,
            .prev = .{ self, other },
            .op = Op.mul,
            .backward = mul_backward,
        };
    }

    pub fn mul_backward(self: *Value) void {
        // self.grad += out.data * out.grad
        self.prev[0].?.grad += self.prev[1].?.data * self.grad;
        // other.grad += self.data * out.grad
        self.prev[1].?.grad += self.prev[0].?.data * self.grad;
    }

    pub fn pow(self: *Value, other: *Value) Value {
        return Value{
            .data = std.math.pow(f64, self.data, other.data),
            .prev = .{ self, other },
            .op = Op.pow,
            .backward = pow_backward,
        };
    }

    pub fn pow_backward(self: *Value) void {
        self.prev[0].?.grad += self.prev[1].?.data * (std.math.pow(f64, self.prev[0].?.data, (self.prev[1].?.data - 1))) * self.grad;
    }

    pub fn tanh(self: *Value) Value {
        const x = self.data;
        const t = (std.math.exp(2 * x) - 1) / (std.math.exp(2 * x) + 1);
        return Value{
            .data = t,
            .prev = .{ self, null },
            .op = Op.tanh,
            .backward = tanh_backward,
        };
    }

    pub fn tanh_backward(self: *Value) void {
        const t = self.data;
        self.prev[0].?.grad += (1 - t * t) * self.grad;
    }

    pub fn exp(self: *Value) Value {
        return Value{
            .data = std.math.exp(self.data),
            .prev = .{ self, null },
            .op = Op.exp,
            .backward = exp_backward,
        };
    }

    pub fn exp_backward(self: *Value) void {
        self.prev[0].?.grad += self.data * self.grad;
    }

    pub fn show(self: *const Value, indent: usize) void {
        std.debug.print("{s: >[1]}Value(label: {[2]s}, op: {[3]s}, data: {[4]d}, grad: {[5]d})\n", .{ "", indent, self.label, @tagName(self.op), self.data, self.grad });

        for (self.prev) |p| {
            if (p) |v| {
                v.show(indent + 2);
            }
        }
    }

    pub fn backprop(self: *Value) !void {
        // topological sort
        // TODO: figure out an appropriate mem allocator
        const allocator = std.heap.page_allocator;
        var visited = std.AutoHashMap(*Value, bool).init(allocator);
        defer visited.deinit();

        var topo = std.ArrayList(*Value).init(allocator);
        defer topo.deinit();

        try build_topo(allocator, self, &visited, &topo);

        self.grad = 1.0;
        var i = topo.items.len;
        while (i > 0) : (i -= 1) {
            var item = topo.items[i - 1];
            item.backward(item);
        }
    }
};

test "gradient is correct when using same var multiple times" {
    var a = Value.init(3.0, "a");
    var b = a.add(&a);
    b.label = "b";
    try b.backprop();
    // b.show(0);

    try std.testing.expect(a.grad == 2);
}

test "2-dim neuron with staged tanh" {
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
    // step-by-step tanh ----------------------------
    var two = Value.init(2.0, "two");
    var n2 = n.mul(&two);
    n2.label = "n2";
    var e = n2.exp();
    e.label = "n2_exp";
    var pos_1 = Value.init(1.0, "pos1");
    pos_1.label = "pos_1";
    var neg_1 = Value.init(-1.0, "neg1");
    neg_1.label = "neg_1";
    var o_above = e.add(&neg_1);
    o_above.label = "e_above";
    var o_below = e.add(&pos_1);
    o_below.label = "e_below";
    var o_below_pow_neg1 = o_below.pow(&neg_1);
    o_below_pow_neg1.label = "e_below_pow_neg1";
    var o = o_above.mul(&o_below_pow_neg1);
    o.label = "o";
    // ----------------------------------------------

    try o.backprop();

    o.show(0);

    try expectApproxEqAbs(o.data, 0.7071067811865476, 1e-12);
}

test "2-dim neuron" {
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

    try o.backprop();
    // o.show(0);

    // check data
    try expectApproxEqAbs(o.data, 0.7071067811865476, 1e-12);
    try expectApproxEqAbs(n.data, 0.8813735870195432, 1e-12);
    try expectApproxEqAbs(x1w1x2w2.data, -6.0, 1e-12);
    try expectApproxEqAbs(x1w1.data, -6.0, 1e-12);
    try expectApproxEqAbs(x2w2.data, 0.0, 1e-12);
    try expectApproxEqAbs(x1.data, 2.0, 1e-12);
    try expectApproxEqAbs(w1.data, -3.0, 1e-12);
    try expectApproxEqAbs(x2.data, 0.0, 1e-12);
    try expectApproxEqAbs(w2.data, 1.0, 1e-12);
    try expectApproxEqAbs(b.data, 6.881373587019543, 1e-12);

    // check grads
    try expectApproxEqAbs(o.grad, 1.0, 1e-12);
    try expectApproxEqAbs(n.grad, 0.5, 1e-12);
    try expectApproxEqAbs(x1w1x2w2.grad, 0.5, 1e-12);
    try expectApproxEqAbs(x1w1.grad, 0.5, 1e-12);
    try expectApproxEqAbs(x2w2.grad, 0.5, 1e-12);
    try expectApproxEqAbs(x1.grad, -1.5, 1e-12);
    try expectApproxEqAbs(w1.grad, 1.0, 1e-12);
    try expectApproxEqAbs(x2.grad, 0.5, 1e-12);
    try expectApproxEqAbs(w2.grad, 0.0, 1e-12);
    try expectApproxEqAbs(b.grad, 0.5, 1e-12);
}
