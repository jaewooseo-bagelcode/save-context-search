//! Tree-sitter query constants for definition extraction.
//!
//! Definition queries extract symbol definitions (functions, classes, structs, etc.)
//! for each supported programming language.

// ============ DEFINITION QUERIES ============

/// Rust definition query - extracts functions, structs, enums, traits, impls, and constants
/// Note: impl_item supports multiple type forms (simple, generic, scoped)
/// Note: function_signature_item captures trait method signatures (without body)
pub const RUST_DEF_QUERY: &str = r#"
(function_item name: (identifier) @function.name) @function
(function_signature_item name: (identifier) @method.name) @method
(struct_item name: (type_identifier) @struct.name) @struct
(enum_item name: (type_identifier) @enum.name) @enum
(trait_item name: (type_identifier) @trait.name) @trait
(impl_item type: (type_identifier) @impl.type) @impl
(impl_item type: (generic_type type: (type_identifier) @impl.type)) @impl
(impl_item type: (scoped_type_identifier name: (type_identifier) @impl.type)) @impl
(const_item name: (identifier) @const.name) @const
"#;

/// TypeScript definition query - works for both .ts and .tsx files
/// Note: TypeScript uses type_identifier for class names
/// Added: lexical_declaration for const/let exports, export_statement wrapping
/// Added: export_statement with arrow_function for React components like `export const Button = () => {}`
pub const TS_DEF_QUERY: &str = r#"
(function_declaration name: (identifier) @function.name) @function
(class_declaration name: (type_identifier) @class.name) @class
(method_definition name: (property_identifier) @method.name) @method
(interface_declaration name: (type_identifier) @interface.name) @interface
(type_alias_declaration name: (type_identifier) @type.name) @type
(enum_declaration name: (identifier) @enum.name) @enum
(lexical_declaration
  (variable_declarator name: (identifier) @const.name)) @const
(export_statement
  declaration: (lexical_declaration
    (variable_declarator
      name: (identifier) @function.name
      value: (arrow_function)))) @function
"#;

/// JavaScript definition query - requires separate parser from TypeScript!
pub const JS_DEF_QUERY: &str = r#"
(function_declaration name: (identifier) @function.name) @function
(class_declaration name: (identifier) @class.name) @class
(method_definition name: (property_identifier) @method.name) @method
"#;

/// C# definition query - includes struct declarations
pub const CS_DEF_QUERY: &str = r#"
(class_declaration name: (identifier) @class.name) @class
(struct_declaration name: (identifier) @struct.name) @struct
(method_declaration name: (identifier) @method.name) @method
(interface_declaration name: (identifier) @interface.name) @interface
(enum_declaration name: (identifier) @enum.name) @enum
(property_declaration name: (identifier) @property.name) @property
"#;

/// Python definition query
/// Note: async def functions are also captured by function_definition in newer tree-sitter-python
/// (async_function_definition was planned but may not exist in all versions)
pub const PY_DEF_QUERY: &str = r#"
(function_definition name: (identifier) @function.name) @function
(class_definition name: (identifier) @class.name) @class
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_constants_not_empty() {
        assert!(!RUST_DEF_QUERY.is_empty());
        assert!(!TS_DEF_QUERY.is_empty());
        assert!(!JS_DEF_QUERY.is_empty());
        assert!(!CS_DEF_QUERY.is_empty());
        assert!(!PY_DEF_QUERY.is_empty());
    }
}
