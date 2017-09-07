CREATE TABLE IF NOT EXISTS inv_values (invariant TEXT, graph TEXT, value FLOAT, UNIQUE(invariant, graph));
CREATE TABLE IF NOT EXISTS prop_values (property TEXT, graph TEXT, value BOOLEAN, UNIQUE(property, graph));
