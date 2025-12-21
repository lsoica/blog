---
title:  "Courses: React Specialisation"
description: >-
  This is a summary of the Coursera React Specialisation Meta course.
author: lso
date:   2025-12-07 11:08:03 +0200
categories: [Blogging, Tutorial]
tags: [react, coursera, specialisation, meta, tutorial]
pin: false
media_subpath: '/posts/20251207'
---
# Intro

To create a new React application, you can use the Create React App (CRA) tool, which sets up a modern React development environment with a single command: npm init react-app . -> npm exec create-react-app . followed by npm start to run the development server.

Virtual DOM versus Real DOM: The Virtual DOM is a lightweight copy of the Real DOM that React uses to optimize updates. When the state of a component changes, React updates the Virtual DOM first, then compares it to the previous version (a process called "diffing") to determine the most efficient way to update the Real DOM.

React diffing algorithm: React's diffing algorithm is a process that compares the current Virtual DOM with the previous version to identify changes. It uses a heuristic approach to minimize the number of updates needed to synchronize the Real DOM with the Virtual DOM, improving performance.

React keys: Keys are unique identifiers assigned to elements in a list when rendering multiple components in React. They help React efficiently update and manage the list by tracking which items have changed, been added, or removed. Keys should be stable, unique, and not change between renders.

React Components: React components are reusable, self-contained pieces of UI that can be defined as either functions (functional components) or classes (class components). They accept inputs called "props" and manage their own state to render dynamic content.

Root Components: In React, the root component is the top-level component that serves as the entry point for rendering the entire application. It is typically rendered into a DOM element with an ID of "root" in the HTML file.

All component names must be capitalized in React. This convention helps distinguish React components from regular HTML elements, which are lowercase.

JSX: JSX (JavaScript XML) is a syntax extension for JavaScript that allows you to write HTML-like code within JavaScript. It makes it easier to create and visualize the structure of React components.

Transpilation: JSX is not valid JavaScript, so it needs to be transpiled (converted) into regular JavaScript using tools like Babel before it can be executed in the browser.

Props: Props (short for "properties") are read-only inputs that are passed from a parent component to a child component in React. They allow you to customize the behavior and appearance of components.

Event handling: JSX vs HTML: In JSX, event handlers are written in camelCase (e.g., onClick) and are passed as functions, whereas in HTML, event handlers are written in lowercase (e.g., onclick) and are typically assigned as strings. The difference is due to JSX being JavaScript-based, allowing for more dynamic and flexible event handling.

Function defined as an expression versus a declaration: A function declaration defines a named function and is hoisted, meaning it can be called before its definition in the code. A function expression defines a function as part of a variable assignment and is not hoisted, so it cannot be called before it is defined. Example of function declaration:

```javascript
function myFunction() {
  // function body
}
```
Example of function expression:
```javascript
const myFunction = function() {
  // function body
}
```

## Components

### Controlled

A controlled component in React is a component that does not maintain its own internal state for form elements (like input, textarea, select). Instead, the state is controlled by React through props. The value of the form element is set by the state of the parent component, and any changes to the form element are handled by event handlers that update the state in the parent component.

### Uncontrolled

An uncontrolled component in React is a component that maintains its own internal state for form elements. The value of the form element is managed by the DOM itself, and React does not control it through props. To access the value of an uncontrolled component, you typically use refs to get the current value from the DOM when needed.

### Function components
Function components are a simpler way to write components that only contain a render method and do not have their own state. They are defined as JavaScript functions that return JSX. Example:

```javascript
function MyComponent(props) {
  return <div>Hello, {props.name}!</div>;
}
```

### Class components
Class components are a more traditional way to write components that can have their own state and lifecycle methods. They are defined as ES6 classes that extend the React.Component class. Example:

```javascript
class MyComponent extends React.Component {
  render() {
    return <div>Hello, {this.props.name}!</div>;
  }
}
```

### Composition with children

Example:

```javascript
function Container(props) {
  return <div className="container">{props.children}</div>;
}
```

And usage:

```javascript
<Container>
  <h1>Hello, World!</h1>
  <p>This is a child element.</p>
</Container>
```

This will render:

```html
<div class="container">
  <h1>Hello, World!</h1>
  <p>This is a child element.</p>
</div>
```

### Spread operator

Usage of the spread operator in React components:

```javascript
const props = { name: 'John', age: 30 };
<MyComponent {...props} />
```

For the MyComponent definition:

```javascript
function MyComponent({ name, age }) {
  return (<div>
    <p>Name: {name}</p>
    <p>Age: {age}</p>
  </div>);
}
```

### Forms

An example of a controller form component:

```javascript
function MyForm() {
  const [value, setValue] = React.useState('');
  const handleChange = (event) => {
    setValue(event.target.value);
  };
  const handleSubmit = (event) => {
    event.preventDefault();
    alert('Submitted value: ' + value);
  };
  return (
    <form onSubmit={handleSubmit}>
      <input type="text" value={value} onChange={handleChange} />
      <button type="submit">Submit</button>
    </form>
  );
}
```

### Reusing behaviour

#### Higher Order Components

HOC is a function that takes a component and returns a new component with added functionality. Example:

```javascript
function withLogging(WrappedComponent) {
  return function(props) {
    console.log('Rendering component with props:', props);
    return <WrappedComponent {...props} />;
  };
}
```

#### Render props

Render props is a technique where a component accepts a function as a prop that returns a React element. Example:

```javascript
function DataFetcher({ render }) {
  const [data, setData] = React.useState(null);
  React.useEffect(() => {
    fetch('/api/data')
      .then(response => response.json())
      .then(data => setData(data));
  }, []);
  return render(data);
}
```

And usage:

```javascript
<DataFetcher render={(data) => (
  data ? <div>Data: {data}</div> : <div>Loading...</div>
)} />
```

## Props

Props are read-only inputs that are passed from a parent component to a child component in React. They allow you to customize the behavior and appearance of components.

### Prop drilling

Prop drilling is the process of passing props from a parent component down to multiple levels of child components, even if some intermediate components do not need the props. This can lead to unnecessary complexity and make the code harder to maintain.

## Children prop

Children prop is a special prop in React that allows you to pass child elements directly into a component. It is accessed via props.children within the component.

## Event handling

Handling events in React involves using camelCase syntax for event names and passing functions as event handlers. Example:

```javascript
function MyButton() {
  const handleClick = () => {
    alert('Button clicked!');
  };
  return <button onClick={handleClick}>Click me</button>;
}
```

## React State

State is a built-in object in React components that allows you to store and manage dynamic data that can change over time. State changes trigger re-renders of the component to reflect the updated data.

## Hooks

Hooks are special functions in React that allow you to use state and other React features in functional components. Hooks can be built-in (like useState, useEffect) or custom-defined.

### useState

useState is a built-in React hook that allows you to add state to functional components. It returns an array with two elements: the current state value and a function to update that state. Example:

```javascript
const [count, setCount] = React.useState(0);
```

### useEffect

useEffect is a built-in React hook that allows you to perform side effects in functional components, such as data fetching, subscriptions, or manually changing the DOM. It takes a function as its first argument and an optional dependency array as its second argument. Example of using useEffect to fetch data:

```javascript
React.useEffect(() => {
  fetch('/api/data')
    .then(response => response.json())
    .then(data => setData(data));
}, []); // Empty dependency array means this effect runs once on mount
```

The dependency array controls when the effect runs. If the array is empty, the effect runs only once when the component mounts. If it contains variables, the effect runs whenever those variables change.

### useContext

useContext is a built-in React hook that allows you to access the value of a context directly within a functional component. A provider component is used to supply the context value to its child components. Example:

```javascript
const MyContext = React.createContext();
function MyProvider({ children }) {
  const value = { /* context value */ };
  return (
    <MyContext.Provider value={value}>
      {children}
    </MyContext.Provider>
  );
}
function MyComponent() {
  const contextValue = React.useContext(MyContext);
  // Use contextValue in the component
}
```

### useReducer

useReducer is a built-in React hook that provides an alternative to useState for managing complex state logic in functional components. It is similar to Redux's reducer pattern. It takes a reducer function and an initial state as arguments and returns the current state and a dispatch function to update the state. Example:

```javascript
const [state, dispatch] = React.useReducer(reducer, initialState);
```

The reducer function takes the current state and an action as arguments and returns the new state based on the action type. Example:
```javascript
function reducer(state, action) {
  switch (action.type) {
    case 'increment':
      return { count: state.count + 1 };
    case 'decrement':
      return { count: state.count - 1 };
    default:
      return state;
  }
}
```

The dispatch function is used to send actions to the reducer to update the state. Example:
```javascript
dispatch({ type: 'increment' });
```

### useRef

useRef is a built-in React hook that allows you to create a mutable reference that persists across renders. It can be used to access DOM elements directly or to store mutable values that do not trigger re-renders when changed. Example of using useRef to access a DOM element:

```javascript
const inputRef = React.useRef(null);
function focusInput() {
  inputRef.current.focus();
}
return <input ref={inputRef} />;
```

### Custom hooks

Custom hooks are user-defined functions that use built-in React hooks to encapsulate and reuse stateful logic across multiple components. They follow the naming convention of starting with "use". Example:

```javascript
function useCounter(initialValue) {
  const [count, setCount] = React.useState(initialValue);
  const increment = () => setCount(count + 1);
  const decrement = () => setCount(count - 1);
  return { count, increment, decrement };
}
```

To use the custom hook in a component:

```javascript
function Counter() {
  const { count, increment, decrement } = useCounter(0);
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={increment}>Increment</button>
      <button onClick={decrement}>Decrement</button>
    </div>
  );
}
```

## Assets

In order to include images in a React application, you can import them directly into your components. Example:

```javascript
import React from 'react';
import myImage from './path/to/image.jpg';
function MyComponent() {
  return <img src={myImage} alt="Description" />;
}
```

### Bundling

To optimize asset loading in a React application, you can use bundling tools like Webpack or Parcel. These tools combine multiple files into a single bundle, reducing the number of HTTP requests and improving load times. They also support features like code splitting and lazy loading to further enhance performance.

## Lists

Lists in React are typically rendered using the map() function to iterate over an array of data and return a list of components. Each item in the list should have a unique "key" prop to help React identify which items have changed, been added, or removed. Example:
```javascript
const items = ['Item 1', 'Item 2', 'Item 3'];
function ItemList() {
  return (
    <ul>
      {items.map((item, index) => (
        <li key={index}>{item}</li>
      ))}
    </ul>
  );
}
```

Keys should be unique and stable, meaning they should not change between renders. Using array indices as keys is not recommended if the list can change, as it can lead to performance issues and unexpected behavior.

### Transforming
Transforming data for lists in React often involves using array methods like map(), filter(), and reduce() to manipulate the data before rendering it. Example of transforming a list of objects to display only certain properties:

```javascript
const users = [
  { id: 1, name: 'Alice', age: 25 },
  { id: 2, name: 'Bob', age: 30 },
  { id: 3, name: 'Charlie', age: 35 }
];
function UserList() {
  return (
    <ul>
      {users.map(user => (
        <li key={user.id}>{user.name} - {user.age} years old</li>
      ))}
    </ul>
  );
}
```

## third party libraries

### Chackra UI

Chakra UI is a popular React component library that provides a set of accessible, reusable, and customizable UI components. It follows the principles of design systems and allows developers to build responsive and visually appealing applications quickly. Chakra UI offers features like theming, dark mode support, and built-in accessibility.

Example of using Chakra UI components:

```javascript
import { Button } from '@chakra-ui/react';
function MyComponent() {
  return <Button colorScheme="teal">Click me</Button>;
}
```

And virtual and horizontal stacks:

```javascript
import { VStack, HStack, Box } from '@chakra-ui/react';
function MyStackComponent() {
  return (
    <VStack spacing={4}>
      <Box bg="tomato" w="100%" p={4} color="white">
        This is a vertical stack item
      </Box>
      <HStack spacing={4}>
        <Box bg="blue" w="100%" p={4} color="white">
          This is a horizontal stack item
        </Box>
        <Box bg="green" w="100%" p={4} color="white">
          This is another horizontal stack item
        </Box>
      </HStack>
    </VStack>
  );
}
```

### Formik

Formik is a popular React library for building and managing forms. It simplifies form handling by providing a set of tools and components to manage form state, validation, and submission. Formik helps reduce boilerplate code and makes it easier to create complex forms with minimal effort.

Example of using Formik to create a simple form:

```javascript
import { Formik, Form, Field, ErrorMessage } from 'formik';
function MyForm() {
  return (
    <Formik
      initialValues={{ name: '' }}
      validate={values => {
        const errors = {};
        if (!values.name) {
          errors.name = 'Required';
        }
        return errors;
      }}
    >
      {({ isSubmitting }) => (
        <Form>
          <Field type="text" name="name" />
          <ErrorMessage name="name" component="div" />
          <button type="submit" disabled={isSubmitting}>
            Submit
          </button>
        </Form>
      )}
    </Formik>
  );
}
```

### Yup

Yup is a JavaScript schema builder for value parsing and validation. It is often used in conjunction with Formik to define validation schemas for forms. Yup allows you to create complex validation rules in a declarative way, making it easier to manage form validation logic.

Example of using Yup to define a validation schema:

```javascript
import * as Yup from 'yup';
const validationSchema = Yup.object().shape({
  name: Yup.string().required('Name is required'),
  age: Yup.number().min(0, 'Age must be a positive number').required('Age is required'),
});
```

And integrating it with Formik:

```javascript
<Formik
  initialValues={{ name: '', age: '' }}
  validationSchema={validationSchema}
>
  {/* form components */}
</Formik>
```

## Testing

### React Testing Library

React Testing Library is a popular testing library for React applications that focuses on testing components from the user's perspective. It provides utilities to render components, query elements, and simulate user interactions. The library encourages writing tests that are more maintainable and less coupled to implementation details.

### Jest

Jest is a widely used JavaScript testing framework that works well with React applications. It provides a simple and powerful way to write unit tests, integration tests, and snapshot tests. Jest comes with built-in features like test runners, assertion libraries, and mocking capabilities, making it easy to set up and run tests for React components.

How to test a React component with React Testing Library and Jest:

```javascript
import { render, screen, fireEvent } from '@testing-library/react';
import MyComponent from './MyComponent';
test('renders MyComponent and handles button click', () => {
  render(<MyComponent />);
  const buttonElement = screen.getByText(/Click me/i);
  expect(buttonElement).toBeInTheDocument();
  fireEvent.click(buttonElement);
  const messageElement = screen.getByText(/Button clicked!/i);
  expect(messageElement).toBeInTheDocument();
});
```