import { Component } from '@angular/core';
import { AddRepositoryComponent } from '../add-repository/add-repository.component';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [AddRepositoryComponent],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css',
})
export class HomeComponent {

}
