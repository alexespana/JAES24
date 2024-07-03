import { Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { AddRepositoryComponent } from './add-repository/add-repository.component';

export const routes: Routes = [
    { path: '', redirectTo: '/home', pathMatch: 'full'},
    { path: 'home', component: HomeComponent },
];
